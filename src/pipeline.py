import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from concurrent.futures import ThreadPoolExecutor
import cv2
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.color.color_detector import detect_dominant_color
from src.logging.logger import log_detection, flush_log
from src.ocr.plate_ocr import read_plate

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MIN_VEHICLE_CONFIDENCE = 0.4
FRAME_SKIP = 3
TRACK_MAX_AGE = 30
OCR_COOLDOWN_FRAMES = 9 

def format_video_timestamp(frame_idx, fps):
    if fps <= 0: return "00:00.0"
    total_seconds = frame_idx / fps
    return f"{int(total_seconds // 60):02d}:{total_seconds % 60:04.1f}"

device = "cuda" if torch.cuda.is_available() else "cpu"
vehicle_model = YOLO("yolov8n.pt").to(device)
plate_model = YOLO("models/yolo_plate/best.pt").to(device)

# --- Supervision Annotators ---
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

_ocr_pool = None
_ocr_pool_engine = None
_pending_futures = []

def _get_ocr_pool(engine_name="EasyOCR"):
    global _ocr_pool, _ocr_pool_engine
    if _ocr_pool is not None and _ocr_pool_engine == engine_name: return _ocr_pool
    if _ocr_pool is not None: _ocr_pool.shutdown(wait=False)
    
    workers = min(os.cpu_count() or 2, 6) if engine_name == "PyTesseract" else 2
    _ocr_pool = ThreadPoolExecutor(max_workers=workers)
    _ocr_pool_engine = engine_name
    return _ocr_pool

class DetectionTracker:
    def __init__(self):
        self.tracks = {} # Swapped to Dictionary for O(1) lookups
        self.tracker = sv.ByteTrack(track_activation_threshold=0.4, lost_track_buffer=30, minimum_matching_threshold=0.3)
        self.global_logged_plates = {}

    def purge_old(self, frame_idx, fps=30.0, force_flush=False):
        active = {}
        for tid, t in self.tracks.items():
            if force_flush or (frame_idx - t["last_seen"] > TRACK_MAX_AGE):
                if t["best_plate"] and t["best_score"] > 0:
                    plate_crop_path = t.get("plate_crop_path") or f"outputs/plate_crops/Processed/frame{t['last_seen']}_track{tid}_processed.jpg"
                    log_detection(
                        t["track_id"], t["vehicle_type"], t["color"], t["best_plate"],
                        t["best_ocr_conf"], t["snapshot_path"], plate_crop_path=plate_crop_path,
                        video_timestamp=t.get("video_timestamp", format_video_timestamp(t["last_seen"], fps))
                    )
            else:
                active[tid] = t
        self.tracks = active

    def get_or_create(self, tracker_id, bbox, vehicle_type, frame_idx):
        if tracker_id in self.tracks:
            t = self.tracks[tracker_id]
            t["prev_bbox"], t["bbox"], t["last_seen"], t["age"] = t["bbox"], bbox, frame_idx, t["age"] + 1
            return t

        track = {
            "track_id": tracker_id, "bbox": bbox, "prev_bbox": None, "vehicle_type": vehicle_type,
            "first_seen": frame_idx, "last_seen": frame_idx, "age": 1, "color": None,
            "best_plate": None, "best_score": 0.0, "best_ocr_conf": 0.0, "max_plate_area": 0,
            "snapshot_path": None, "last_ocr_frame": -999, "global_plate_bbox": None,
            "video_timestamp": "", "plate_crop_path": ""
        }
        self.tracks[tracker_id] = track
        return track

    def update_plate(self, track, plate_text, ocr_conf, plate_area, snapshot_path, frame_idx, video_timestamp=""):
        effective_conf = max(ocr_conf, 0.01) if (plate_text and len(plate_text.strip()) > 0) else ocr_conf
        score = plate_area * effective_conf

        if not (any(c.isalpha() for c in plate_text) and any(c.isdigit() for c in plate_text)): score *= 0.1
        score *= (len(plate_text.replace(" ", "")) / 7.0)

        is_first = track.get("best_plate") is None
        has_more_data = sum(c.isalnum() for c in plate_text) > sum(c.isalnum() for c in (track.get("best_plate") or ""))
        override = (ocr_conf > 0.10 and track.get("best_ocr_conf", 0) <= 0.01) or (score > track.get("best_score", 0))

        if is_first or override or has_more_data:
            track["best_plate"], track["best_score"], track["best_ocr_conf"] = plate_text, max(score, track.get("best_score", 0)), ocr_conf
            track["snapshot_path"], track["video_timestamp"] = snapshot_path, video_timestamp
            track["plate_crop_path"] = f"outputs/plate_crops/Processed/frame{frame_idx}_track{track['track_id']}_processed.jpg"

detection_tracker = DetectionTracker()

# ─── Async OCR Worker & Harvesting ────────────────────────────────

def _ocr_worker(plate_crop, vehicle_crop, ocr_engine, track_id, plate_area, frame_idx, fps=30.0):
    plate_text, ocr_conf, engine, processed_crop = read_plate(plate_crop, ocr_engine)
    cv2.imwrite(f"outputs/plate_crops/Raw/frame{frame_idx}_track{track_id}_raw.jpg", plate_crop)
    if processed_crop is not None: cv2.imwrite(f"outputs/plate_crops/Processed/frame{frame_idx}_track{track_id}_processed.jpg", processed_crop)
    
    snapshot_path = f"outputs/snapshots/frame{frame_idx}_{plate_text.replace(' ', '_')}.jpg" if plate_text and ocr_conf > 0 else None
    if snapshot_path: cv2.imwrite(snapshot_path, vehicle_crop)
    return plate_text, ocr_conf, track_id, plate_area, frame_idx, snapshot_path, format_video_timestamp(frame_idx, fps)

def _apply_ocr_result(future):
    try:
        text, conf, track_id, area, frame_idx, snap_path, v_time = future.result()
        if text and conf > 0 and snap_path:
            track = detection_tracker.tracks.get(track_id)
            if track:
                track["max_plate_area"] = max(track.get("max_plate_area", 0), area)
                detection_tracker.update_plate(track, text, conf, area, snap_path, frame_idx, video_timestamp=v_time)
                log_detection(track_id, track["vehicle_type"], track["color"], track["best_plate"], track["best_ocr_conf"], track["snapshot_path"], plate_crop_path=track.get("plate_crop_path", ""), video_timestamp=track.get("video_timestamp", ""))
    except Exception as e: print(f"[OCR] Worker failed: {e}")

def _harvest_ocr_results(pending_futures):
    still_pending = [f for f in pending_futures if not f.done()]
    for f in [f for f in pending_futures if f.done()]: _apply_ocr_result(f)
    pending_futures[:] = still_pending

def drain_pending_ocr(pending_futures=None):
    for f in (pending_futures if pending_futures is not None else _pending_futures): _apply_ocr_result(f)
    if pending_futures is not None: pending_futures.clear()
    else: _pending_futures.clear()
    flush_log()

# ─── Batch Processing Pipeline ───────────────────────────────────

def process_batch(frames_batch, frame_indices, ocr_engine, ocr_pool, pending_futures, fps=30.0):
    _harvest_ocr_results(pending_futures)
    resized_frames = [cv2.resize(f, (1920, int(f.shape[0] * (1920 / f.shape[1])))) if f.shape[1] > 1920 else f for f in frames_batch]
    vehicle_results_batch = vehicle_model(resized_frames, verbose=False, conf=MIN_VEHICLE_CONFIDENCE)
    processed_frames = []

    for idx, (frame, frame_idx, vehicle_results) in enumerate(zip(resized_frames, frame_indices, vehicle_results_batch)):
        detection_tracker.purge_old(frame_idx, fps=fps)

        # Supervision: Filter classes & update ByteTrack
        detections = sv.Detections.from_ultralytics(vehicle_results)
        detections = detections[np.isin(detections.class_id, list(VEHICLE_CLASSES.keys()))]
        tracked_detections = detection_tracker.tracker.update_with_detections(detections)

        valid_vehicles, labels = [], []

        for bbox, class_id, tracker_id in zip(tracked_detections.xyxy, tracked_detections.class_id, tracked_detections.tracker_id):
            vehicle_crop = sv.crop_image(image=frame, xyxy=bbox)
            if vehicle_crop.size == 0: continue

            track = detection_tracker.get_or_create(int(tracker_id), bbox, VEHICLE_CLASSES[int(class_id)], frame_idx)
            if track["color"] is None: track["color"] = detect_dominant_color(vehicle_crop)

            # Build label for Supervision annotator
            labels.append(" ".join([p for p in [track["vehicle_type"], track["color"], track["best_plate"]] if p]))

            if (len(track.get("best_plate") or "") >= 6 and track.get("best_ocr_conf", 0) > 0.8): continue
            valid_vehicles.append((track, vehicle_crop, bbox))

        # Batched plate detection pass
        if valid_vehicles:
            plate_results_batch = plate_model([v[1] for v in valid_vehicles], verbose=False, conf=0.5)

            for (track, vehicle_crop, v_bbox), p_res in zip(valid_vehicles, plate_results_batch):
                p_det = sv.Detections.from_ultralytics(p_res)
                if len(p_det) == 0: continue

                # Get Highest Confidence Plate
                best_idx = np.argmax(p_det.confidence)
                px1, py1, px2, py2 = map(int, p_det.xyxy[best_idx])
                plate_area = p_det.area[best_idx]
                
                # Global mapping
                track["global_plate_bbox"] = (v_bbox[0] + px1, v_bbox[1] + py1, v_bbox[0] + px2, v_bbox[1] + py2)

                plate_grew = plate_area > track.get("max_plate_area", 0) * 1.1
                if plate_grew or track.get("best_score", 0) < 5000:
                    if (frame_idx - track.get("last_ocr_frame", -999) < OCR_COOLDOWN_FRAMES) and not plate_grew: continue

                    pad_x, pad_y = int((px2 - px1) * 0.08), int((py2 - py1) * 0.15)
                    plate_crop = vehicle_crop[max(0, py1 - pad_y):min(vehicle_crop.shape[0], py2 + pad_y), max(0, px1 - pad_x):min(vehicle_crop.shape[1], px2 + pad_x)]

                    if plate_crop.size > 0 and cv2.Laplacian(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() >= 50.0:
                        track["last_ocr_frame"] = frame_idx
                        pending_futures.append(ocr_pool.submit(_ocr_worker, plate_crop.copy(), vehicle_crop.copy(), ocr_engine, track["track_id"], plate_area, frame_idx, fps))

        # Supervision: Draw all bounding boxes and labels at once
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

        # Draw the internal red plate bounding box
        for track in detection_tracker.tracks.values():
            if frame_idx - track["last_seen"] <= 5 and track.get("global_plate_bbox"):
                gpx1, gpy1, gpx2, gpy2 = map(int, track["global_plate_bbox"])
                cv2.rectangle(annotated_frame, (gpx1, gpy1), (gpx2, gpy2), (0, 0, 255), 2)

        processed_frames.append(annotated_frame)
    return processed_frames

def process_frame(frame, frame_idx, ocr_engine="EasyOCR", fps=30.0):
    global _pending_futures
    return process_batch([frame], [frame_idx], ocr_engine, _get_ocr_pool(ocr_engine), _pending_futures, fps)[0]
