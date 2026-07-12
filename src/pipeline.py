import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import itertools
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
from ultralytics import YOLO
from src.color.color_detector import detect_dominant_color
from src.logging.logger import log_detection, flush_log
from src.ocr.plate_ocr import read_plate

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MIN_VEHICLE_CONFIDENCE = 0.4
FRAME_SKIP = 3

def format_video_timestamp(frame_idx, fps):
    """Convert a frame index and FPS to a video timecode string (MM:SS.s)."""
    if fps <= 0:
        return "00:00.0"
    total_seconds = frame_idx / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:04.1f}"

device = "cuda" if torch.cuda.is_available() else "cpu"
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("models/yolo_plate/best.pt")
vehicle_model.to(device)
plate_model.to(device)

TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_AGE = 30

# Async OCR Configuration Cooldown
OCR_COOLDOWN_FRAMES = 9  # Don't re-OCR the same track within 9 processed frames

# Legacy Globals for backward compatibility (in case process_frame is used directly)
_ocr_pool = None
_ocr_pool_engine = None
_pending_futures = []

def _get_ocr_pool(engine_name="EasyOCR"):
    global _ocr_pool, _ocr_pool_engine
    if _ocr_pool is not None and _ocr_pool_engine == engine_name:
        return _ocr_pool
    if _ocr_pool is not None:
        _ocr_pool.shutdown(wait=False)
    if engine_name == "PyTesseract":
        workers = min(os.cpu_count() or 2, 6)
    else:
        workers = 2
    _ocr_pool = ThreadPoolExecutor(max_workers=workers)
    _ocr_pool_engine = engine_name
    return _ocr_pool

class DetectionTracker:
    """
    IoU-based object tracker maintaining vehicle identities and best plate states across frames.
    """
    _id_counter = itertools.count(1)

    def __init__(self):
        self.tracks = []
        self.global_logged_plates = {}

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def purge_old(self, frame_idx, fps=30.0, force_flush=False):
        """Sweeps old tracks that haven't been seen for TRACK_MAX_AGE frames and logs them."""
        active = []
        for t in self.tracks:
            # If track hasn't been seen recently, or we are forcing stream termination flush
            if force_flush or (frame_idx - t["last_seen"] > TRACK_MAX_AGE):
                if t["best_plate"] and t["best_score"] > 0:
                    plate_crop_path = t.get("plate_crop_path")
                    if not plate_crop_path:
                        plate_crop_path = f"outputs/plate_crops/Processed/frame{t['last_seen']}_track{t['track_id']}_processed.jpg"
                    log_detection(
                        t["track_id"],
                        t["vehicle_type"],
                        t["color"],
                        t["best_plate"],
                        t["best_ocr_conf"],
                        t["snapshot_path"],
                        plate_crop_path=plate_crop_path,
                        video_timestamp=t.get("video_timestamp", format_video_timestamp(t["last_seen"], fps))
                    )
            else:
                active.append(t)
        self.tracks = active

    def match_or_create(self, bbox, vehicle_type, frame_idx):
        # Strategy 1: Match by IoU overlap with an existing track
        best_iou = 0.0
        best_track = None
        for track in self.tracks:
            if track["vehicle_type"] == vehicle_type:
                iou = self._iou(track["bbox"], bbox)
                if iou >= TRACK_IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_track = track

        if best_track:
            best_track["prev_bbox"] = best_track["bbox"]
            best_track["bbox"] = bbox
            best_track["last_seen"] = frame_idx
            best_track["age"] += 1
            return best_track

        # Create new track
        track = {
            "track_id": next(self._id_counter),
            "bbox": bbox,
            "prev_bbox": None,
            "vehicle_type": vehicle_type,
            "first_seen": frame_idx,
            "last_seen": frame_idx,
            "age": 1,
            "color": None,
            "best_plate": None,
            "best_score": 0.0,
            "best_ocr_conf": 0.0,
            "max_plate_area": 0,
            "snapshot_path": None,
            "last_ocr_frame": -999,
            "global_plate_bbox": None,
            "video_timestamp": "",
            "plate_crop_path": "",
            "structural_state": {}
        }
        self.tracks.append(track)
        return track

    def get_max_displacement(self):
        """Calculates the maximum displacement center-distance of any active vehicle between frames."""
        displacements = []
        for track in self.tracks:
            if track.get("prev_bbox") is not None:
                x1, y1, x2, y2 = track["bbox"]
                px1, py1, px2, py2 = track["prev_bbox"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                disp = math.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                displacements.append(disp)
        return max(displacements) if displacements else 0.0

    def update_plate(self, track, plate_text, ocr_conf, plate_area, snapshot_path, frame_idx, video_timestamp=""):
        """Update if the new read is better, using temporal voting to prevent single-frame misreads."""
        # Reject extremely low confidence reads early to prevent noise/garbage from registering
        if ocr_conf < 0.15:
            return False

        effective_conf = max(ocr_conf, 0.01) if (plate_text and len(plate_text.strip()) > 0) else ocr_conf
        score = plate_area * effective_conf

        # Heuristic 1: Malaysian plates almost always have both letters and numbers.
        has_letters = any(c.isalpha() for c in plate_text)
        has_numbers = any(c.isdigit() for c in plate_text)
        if not (has_letters and has_numbers):
            score *= 0.1
            
        # Heuristic 2: Favor longer reads over short partial reads.
        compact_len = len(plate_text.replace(" ", ""))
        score *= (compact_len / 7.0)

        # ── Temporal Voting ──
        # Track how many times each unique text has been read across frames.
        # This prevents a single high-confidence misread from overwriting a correct plate.
        votes = track.setdefault("plate_votes", {})
        compact_text = plate_text.replace(" ", "")
        votes[compact_text] = votes.get(compact_text, 0) + 1

        # Decide if we should update:
        is_first_read = track.get("best_plate") is None
        current_best_compact = (track.get("best_plate") or "").replace(" ", "")

        has_high_conf_override = (ocr_conf > 0.10 and track.get("best_ocr_conf", 0) <= 0.01)
        has_better_score = score > track.get("best_score", 0)

        # Voting-aware promotion logic:
        # - First read: always accept
        # - Same text as current best: update if better score
        # - Different text: only accept if voted >=2 times OR has significantly better score (2x)
        if is_first_read or has_high_conf_override:
            should_update = True
        elif compact_text == current_best_compact:
            should_update = has_better_score
        else:
            vote_count = votes.get(compact_text, 0)
            should_update = (vote_count >= 2 and has_better_score) or (score > track.get("best_score", 0) * 2.0)

        if should_update:
            track["best_plate"] = plate_text
            track["best_score"] = max(score, track.get("best_score", 0)) 
            track["best_ocr_conf"] = ocr_conf
            track["snapshot_path"] = snapshot_path
            track["video_timestamp"] = video_timestamp
            track["plate_crop_path"] = f"outputs/plate_crops/Processed/frame{frame_idx}_track{track['track_id']}_processed.jpg"
        return should_update

    def find_by_id(self, track_id):
        for track in self.tracks:
            if track["track_id"] == track_id:
                return track
        return None

    def flush_all(self):
        self.tracks.clear()
        self.global_logged_plates.clear()

detection_tracker = DetectionTracker()

# ─── Async OCR Worker & Harvesting ────────────────────────────────

def _ocr_worker(plate_crop, vehicle_crop, ocr_engine, track_id, plate_area, frame_idx, fps=30.0):
    """Runs OCR + disk I/O in a background thread."""
    plate_text, ocr_conf, engine, processed_crop = read_plate(plate_crop, ocr_engine)

    raw_path = f"outputs/plate_crops/Raw/frame{frame_idx}_track{track_id}_raw.jpg"
    proc_path = f"outputs/plate_crops/Processed/frame{frame_idx}_track{track_id}_processed.jpg"
    cv2.imwrite(raw_path, plate_crop)
    if processed_crop is not None:
        cv2.imwrite(proc_path, processed_crop)

    snapshot_path = None
    if plate_text and ocr_conf > 0:
        snapshot_path = f"outputs/snapshots/frame{frame_idx}_{plate_text.replace(' ', '_')}.jpg"
        cv2.imwrite(snapshot_path, vehicle_crop)

    video_timestamp = format_video_timestamp(frame_idx, fps)
    return plate_text, ocr_conf, track_id, plate_area, frame_idx, snapshot_path, video_timestamp

def _apply_ocr_result(future):
    """Process a single completed OCR future."""
    try:
        text, conf, track_id, plate_area, frame_idx_ocr, snapshot_path, video_timestamp = future.result()
        print(f"[OCR Harvest] Track {track_id}: Text='{text}', Conf={conf:.3f}, Snapshot={snapshot_path}")
        if text and conf > 0 and snapshot_path:
            track = detection_tracker.find_by_id(track_id)
            if track:
                track["max_plate_area"] = max(track.get("max_plate_area", 0), plate_area)
                updated = detection_tracker.update_plate(track, text, conf, plate_area, snapshot_path, frame_idx_ocr, video_timestamp=video_timestamp)
                
                # Real-time write mapping so dashboard updates immediately
                if updated:
                    log_detection(
                        track["track_id"],
                        track["vehicle_type"],
                        track["color"],
                        track["best_plate"],
                        track["best_ocr_conf"],
                        track["snapshot_path"],
                        plate_crop_path=track.get("plate_crop_path", ""),
                        video_timestamp=track.get("video_timestamp", "")
                    )
    except Exception as e:
        print(f"[OCR] Worker failed: {e}")

def _harvest_ocr_results(pending_futures):
    """Non-blocking sweep of finished OCR futures."""
    still_pending = []
    for future in pending_futures:
        if future.done():
            _apply_ocr_result(future)
        else:
            still_pending.append(future)
    pending_futures[:] = still_pending

def drain_pending_ocr(pending_futures=None):
    """Wait for all in-flight OCR tasks to finish."""
    global _pending_futures
    futures_list = pending_futures if pending_futures is not None else _pending_futures
    for future in futures_list:
        _apply_ocr_result(future)
    futures_list.clear()
    flush_log()

# ─── Drawing overlays ──────────────────────────────────────────────

def _draw_overlay(frame, track):
    x1, y1, x2, y2 = track["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    parts = [track["vehicle_type"]]
    if track["color"]:
        parts.append(track["color"])
    if track["best_plate"]:
        parts.append(track["best_plate"])
    cv2.putText(frame, " ".join(parts), (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw plate global bounding box in red inside the green vehicle box
    g_plate = track.get("global_plate_bbox")
    if g_plate:
        gpx1, gpy1, gpx2, gpy2 = g_plate
        cv2.rectangle(frame, (gpx1, gpy1), (gpx2, gpy2), (0, 0, 255), 2)

# ─── Batch Processing Pipeline ───────────────────────────────────

def process_batch(frames_batch, frame_indices, ocr_engine, ocr_pool, pending_futures, fps=30.0):
    """Processes a micro-batch of frames together using batched vehicle detection."""
    # 1. Harvest finished OCR tasks
    _harvest_ocr_results(pending_futures)

    # 2. Downscale frames if they exceed 1080p width
    resized_frames = []
    for frame in frames_batch:
        h, w = frame.shape[:2]
        if w > 1920:
            scale = 1920 / w
            resized_frames.append(cv2.resize(frame, (1920, int(h * scale))))
        else:
            resized_frames.append(frame)

    # 3. Batch vehicle detection
    vehicle_results_batch = vehicle_model(resized_frames, verbose=False, conf=MIN_VEHICLE_CONFIDENCE)

    processed_frames = []

    # 4. Sequentially track vehicles and check crops per frame
    for idx, (frame, frame_idx, vehicle_results) in enumerate(zip(resized_frames, frame_indices, vehicle_results_batch)):
        detection_tracker.purge_old(frame_idx, fps=fps)

        valid_vehicles = []
        for box in vehicle_results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            vehicle_type = VEHICLE_CLASSES[cls_id]
            track = detection_tracker.match_or_create((x1, y1, x2, y2), vehicle_type, frame_idx)

            # Color analyzer on vehicle crop
            if track["color"] is None:
                track["color"] = detect_dominant_color(vehicle_crop)

            # Cooldown / confident plate read skip check
            best_plate = track.get("best_plate") or ""
            best_conf = track.get("best_ocr_conf") or 0.0
            is_confident = (len(best_plate) >= 6 and best_conf > 0.8) or (len(best_plate) >= 4 and best_conf > 0.90)
            if is_confident:
                continue

            valid_vehicles.append((track, vehicle_crop, (x1, y1, x2, y2)))

        # 5. Batched license plate detection pass inside vehicle bounding boxes
        if valid_vehicles:
            crops = [v[1] for v in valid_vehicles]
            plate_results_batch = plate_model(crops, verbose=False, conf=0.5)

            for (track, vehicle_crop, vehicle_bbox), plate_results in zip(valid_vehicles, plate_results_batch):
                if len(plate_results.boxes) > 0:
                    best_plate_box = max(plate_results.boxes, key=lambda p: float(p.conf[0]))

                    if float(best_plate_box.conf[0]) >= 0.5:
                        px1, py1, px2, py2 = map(int, best_plate_box.xyxy[0])
                        plate_area = (px2 - px1) * (py2 - py1)

                        # Project local plate coordinates back to global coordinates
                        vx1, vy1, _, _ = vehicle_bbox
                        gpx1 = vx1 + px1
                        gpy1 = vy1 + py1
                        gpx2 = vx1 + px2
                        gpy2 = vy1 + py2
                        track["global_plate_bbox"] = (gpx1, gpy1, gpx2, gpy2)

                        # Gate 1: Check if plate size is larger or we lack a solid read
                        plate_grew = plate_area > track.get("max_plate_area", 0) * 1.1
                        if plate_grew or track.get("best_score", 0) < 5000:
                            
                            # Gate 2: Cooldown check
                            recently_ocrd = frame_idx - track.get("last_ocr_frame", -999) < OCR_COOLDOWN_FRAMES
                            if recently_ocrd and not plate_grew:
                                continue

                            # Crop the plate with padding
                            pad_x = int((px2 - px1) * 0.08)
                            pad_y = int((py2 - py1) * 0.15)
                            px1_pad, py1_pad = max(0, px1 - pad_x), max(0, py1 - pad_y)
                            px2_pad, py2_pad = min(vehicle_crop.shape[1], px2 + pad_x), min(vehicle_crop.shape[0], py2 + pad_y)
                            plate_crop = vehicle_crop[py1_pad:py2_pad, px1_pad:px2_pad]

                            if plate_crop.size > 0 and plate_crop.shape[0] >= 5 and plate_crop.shape[1] >= 10:
                                # Aspect ratio gate: Malaysian plates are ~3:1 to 5:1 (width:height)
                                aspect = plate_crop.shape[1] / plate_crop.shape[0]
                                if aspect < 1.5 or aspect > 7.0:
                                    continue

                                # Adaptive sharpness check (scales with crop resolution)
                                gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                variance = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                                crop_pixels = plate_crop.shape[0] * plate_crop.shape[1]
                                adaptive_threshold = 50.0 * (crop_pixels / 3000.0)
                                adaptive_threshold = max(30.0, min(adaptive_threshold, 200.0))
                                
                                if variance < adaptive_threshold:
                                    continue

                                # Submit OCR task to background ThreadPoolExecutor
                                track["last_ocr_frame"] = frame_idx
                                future = ocr_pool.submit(
                                    _ocr_worker,
                                    plate_crop.copy(),
                                    vehicle_crop.copy(),
                                    ocr_engine,
                                    track["track_id"],
                                    plate_area,
                                    frame_idx,
                                    fps
                                )
                                pending_futures.append(future)

        # Draw overlays
        for track in detection_tracker.tracks:
            if frame_idx - track["last_seen"] <= 5:
                _draw_overlay(frame, track)

        processed_frames.append(frame)

    return processed_frames

# Legacy fallback process_frame
def process_frame(frame, frame_idx, ocr_engine="EasyOCR", fps=30.0):
    global _pending_futures
    pool = _get_ocr_pool(ocr_engine)
    res = process_batch([frame], [frame_idx], ocr_engine, pool, _pending_futures, fps)
    return res[0]
