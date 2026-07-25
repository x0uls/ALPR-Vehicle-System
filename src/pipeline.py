import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import itertools
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from src.color.color_detector import detect_dominant_color
from src.logging.logger import log_detection, flush_log
from src.ocr.plate_ocr import read_plate

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MIN_VEHICLE_CONFIDENCE = 0.4
TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_AGE = 30
OCR_COOLDOWN_FRAMES = 9

device = "cuda" if torch.cuda.is_available() else "cpu"
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("models/yolo_plate/best.pt")
vehicle_model.to(device)
plate_model.to(device)


def format_video_timestamp(frame_idx, fps):
    if fps <= 0:
        return "00:00.0"
    m, s = divmod(frame_idx / fps, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _compute_iou(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / float(union) if union > 0 else 0.0


def _log_track(track_dict, engine_name, plate_crop_path="", video_timestamp="", log_path="outputs/logs/detections_easyocr.csv"):
    eng_data = track_dict["engines"][engine_name]
    log_detection(
        track_dict["track_id"],
        track_dict["vehicle_type"],
        track_dict["color"],
        eng_data.get("best_plate"),
        eng_data.get("best_ocr_conf", 0.0),
        eng_data.get("snapshot_path"),
        plate_crop_path=plate_crop_path or eng_data.get("plate_crop_path", ""),
        video_timestamp=video_timestamp or eng_data.get("video_timestamp", ""),
        log_path=log_path
    )


class DetectionTracker:
    """Unified vehicle tracker maintaining single vehicle identities with nested per-OCR-engine results."""
    def __init__(self):
        self._id_counter = itertools.count(1)
        self.tracks = []

    def purge_old(self, frame_idx, frames_per_second=30.0, force_flush=False):
        active = []
        for t in self.tracks:
            if force_flush or (frame_idx - t["last_seen"] > TRACK_MAX_AGE):
                for eng_name, log_p in [("EasyOCR", "outputs/logs/detections_easyocr.csv"), ("PyTesseract", "outputs/logs/detections_pytesseract.csv")]:
                    eng_data = t["engines"][eng_name]
                    if eng_data.get("plate_crop_path") or (eng_data.get("best_plate") and eng_data.get("best_score", 0) > 0):
                        crop_p = eng_data.get("plate_crop_path") or f"outputs/plate_crops/Processed/frame{t['last_seen']}_track{t['track_id']}_{eng_name.lower()}.jpg"
                        v_time = eng_data.get("video_timestamp") or format_video_timestamp(t["last_seen"], frames_per_second)
                        _log_track(t, eng_name, plate_crop_path=crop_p, video_timestamp=v_time, log_path=log_p)
            else:
                active.append(t)
        self.tracks = active

    def match_or_create(self, bbox, vehicle_type, frame_idx):
        best_iou, best_track = 0.0, None
        for track in self.tracks:
            if track["vehicle_type"] == vehicle_type:
                iou = _compute_iou(track["bbox"], bbox)
                if iou >= TRACK_IOU_THRESHOLD and iou > best_iou:
                    best_iou, best_track = iou, track

        if best_track:
            best_track["prev_bbox"] = best_track["bbox"]
            best_track["bbox"] = bbox
            best_track["last_seen"] = frame_idx
            best_track["age"] += 1
            return best_track

        t_id = next(self._id_counter)
        track = {
            "track_id": t_id,
            "bbox": bbox,
            "prev_bbox": None,
            "vehicle_type": vehicle_type,
            "first_seen": frame_idx,
            "last_seen": frame_idx,
            "age": 1,
            "color": None,
            "local_plate_bbox": None,
            "engines": {
                "EasyOCR": {
                    "best_plate": None,
                    "best_score": 0.0,
                    "best_ocr_conf": 0.0,
                    "snapshot_path": None,
                    "plate_crop_path": "",
                    "video_timestamp": "",
                    "last_ocr_frame": -999,
                    "max_plate_area": 0,
                    "plate_votes": {},
                },
                "PyTesseract": {
                    "best_plate": None,
                    "best_score": 0.0,
                    "best_ocr_conf": 0.0,
                    "snapshot_path": None,
                    "plate_crop_path": "",
                    "video_timestamp": "",
                    "last_ocr_frame": -999,
                    "max_plate_area": 0,
                    "plate_votes": {},
                },
            }
        }
        self.tracks.append(track)
        return track

    def get_max_displacement(self):
        displacements = [
            np.hypot((t["bbox"][0] + t["bbox"][2] - t["prev_bbox"][0] - t["prev_bbox"][2]) / 2,
                     (t["bbox"][1] + t["bbox"][3] - t["prev_bbox"][1] - t["prev_bbox"][3]) / 2)
            for t in self.tracks if t.get("prev_bbox") is not None
        ]
        return max(displacements, default=0.0)

    @staticmethod
    def _compute_plate_quality_score(plate_text, ocr_conf, plate_area):
        effective_conf = max(ocr_conf, 0.01) if (plate_text and len(plate_text.strip()) > 0) else ocr_conf
        score = plate_area * effective_conf
        if not (any(c.isalpha() for c in plate_text) and any(c.isdigit() for c in plate_text)):
            score *= 0.1
        score *= (len(plate_text.replace(" ", "")) / 7.0)
        return score

    def update_plate(self, track_dict, engine_name, plate_text, ocr_conf, plate_area, snapshot_path, frame_idx, video_timestamp="", crop_path=""):
        eng_data = track_dict["engines"][engine_name]
        score = self._compute_plate_quality_score(plate_text or "", ocr_conf, plate_area)
        compact_text = (plate_text or "").replace(" ", "")

        if compact_text:
            plate_votes = eng_data.setdefault("plate_votes", {})
            plate_votes[compact_text] = plate_votes.get(compact_text, 0) + 1

        curr_best = (eng_data.get("best_plate") or "").replace(" ", "")
        should_replace = (
            eng_data.get("best_plate") is None
            or not eng_data.get("plate_crop_path")
            or (ocr_conf > 0.10 and eng_data.get("best_ocr_conf", 0) <= 0.01)
            or (compact_text and compact_text == curr_best and score > eng_data.get("best_score", 0))
            or (compact_text and plate_votes.get(compact_text, 0) >= 2 and score > eng_data.get("best_score", 0))
            or (score > eng_data.get("best_score", 0) * 2.0)
        )

        if should_replace:
            if plate_text and len(plate_text.strip()) > 0:
                eng_data["best_plate"] = plate_text
                eng_data["best_score"] = max(score, eng_data.get("best_score", 0))
                eng_data["best_ocr_conf"] = ocr_conf
            eng_data["snapshot_path"] = snapshot_path
            eng_data["video_timestamp"] = video_timestamp
            eng_data["plate_crop_path"] = crop_path or f"outputs/plate_crops/Processed/frame{frame_idx}_track{track_dict['track_id']}_{engine_name.lower()}.jpg"
            return True
        return False

    def find_by_id(self, track_id):
        for t in self.tracks:
            if t["track_id"] == track_id:
                return t
        return None

    def flush_all(self):
        self.tracks.clear()


# Single unified vehicle tracker instance
vehicle_tracker = DetectionTracker()


def _ocr_worker(plate_crop_image, vehicle_crop_image, ocr_engine_name, track_id, plate_pixel_area, frame_idx, fps=30.0):
    plate_text, ocr_conf, engine, processed_crop = read_plate(plate_crop_image, ocr_engine_name)

    os.makedirs("outputs/plate_crops/Processed", exist_ok=True)
    processed_crop_path = f"outputs/plate_crops/Processed/frame{frame_idx}_track{track_id}_{ocr_engine_name.lower()}.jpg"
    if processed_crop is not None and processed_crop.size > 0:
        cv2.imwrite(processed_crop_path, processed_crop)
    else:
        cv2.imwrite(processed_crop_path, plate_crop_image)

    os.makedirs("outputs/snapshots", exist_ok=True)
    safe_text = (plate_text or "no_read").replace(' ', '_')
    snapshot_path = f"outputs/snapshots/frame{frame_idx}_track{track_id}_{safe_text}.jpg"
    cv2.imwrite(snapshot_path, vehicle_crop_image)

    video_timestamp = format_video_timestamp(frame_idx, fps)
    return ocr_engine_name, plate_text, ocr_conf, track_id, plate_pixel_area, frame_idx, snapshot_path, video_timestamp, processed_crop_path


def _apply_ocr_result(ocr_future):
    try:
        engine_name, text, conf, track_id, plate_pixel_area, frame_idx_ocr, snapshot_path, video_timestamp, crop_path = ocr_future.result()
        track = vehicle_tracker.find_by_id(track_id)
        if track:
            eng_data = track["engines"][engine_name]
            eng_data["max_plate_area"] = max(eng_data.get("max_plate_area", 0), plate_pixel_area)

            # Always save crop and snapshot paths so preprocessed image is accessible in UI
            if not eng_data.get("plate_crop_path"):
                eng_data["plate_crop_path"] = crop_path
            if not eng_data.get("snapshot_path"):
                eng_data["snapshot_path"] = snapshot_path

            if vehicle_tracker.update_plate(track, engine_name, text, conf, plate_pixel_area, snapshot_path, frame_idx_ocr, video_timestamp=video_timestamp, crop_path=crop_path):
                log_path = f"outputs/logs/detections_{engine_name.lower()}.csv"
                _log_track(track, engine_name, plate_crop_path=crop_path, video_timestamp=video_timestamp, log_path=log_path)
    except Exception as e:
        print(f"[OCR] Worker failed: {e}")


def _harvest_ocr_results(pending_ocr_futures):
    unresolved = []
    for f in pending_ocr_futures:
        if f.done():
            _apply_ocr_result(f)
        else:
            unresolved.append(f)
    pending_ocr_futures[:] = unresolved


def drain_pending_ocr(pending_ocr_futures=None):
    futures_list = pending_ocr_futures if pending_ocr_futures is not None else []
    for f in futures_list:
        _apply_ocr_result(f)
    futures_list.clear()
    flush_log(log_path="outputs/logs/detections_easyocr.csv")
    flush_log(log_path="outputs/logs/detections_pytesseract.csv")


def _draw_overlay(frame, track_dict, model_theme="EasyOCR"):
    x1, y1, x2, y2 = track_dict["bbox"]
    w, h = x2 - x1, y2 - y1
    primary_color = (255, 0, 200) if model_theme == "PyTesseract" else (255, 200, 0)
    eng_data = track_dict["engines"][model_theme]

    cv2.rectangle(frame, (x1, y1), (x2, y2), primary_color, 2)
    corner_len = min(22, max(8, int(min(w, h) * 0.2)))
    for (px, py), (dx, dy) in [((x1, y1), (1, 1)), ((x2, y1), (-1, 1)), ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))]:
        cv2.line(frame, (px, py), (px + dx * corner_len, py), primary_color, 3)
        cv2.line(frame, (px, py), (px, py + dy * corner_len), primary_color, 3)

    parts = [f"#{track_dict['track_id']}", track_dict["vehicle_type"].capitalize()]
    if track_dict.get("color"):
        parts.append(track_dict["color"].capitalize())
    plate_text, conf = eng_data.get("best_plate"), eng_data.get("best_ocr_conf", 0.0)
    parts.append(f"• {plate_text} ({int(conf * 100)}%)" if plate_text else "• Scanning...")

    label_str = " " + " ".join(parts) + " "
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
    (tw, th), _ = cv2.getTextSize(label_str, font, scale, thick)
    badge_y1, badge_y2 = max(0, y1 - th - 12), y1
    badge_x2 = min(frame.shape[1], x1 + tw + 10)

    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), (15, 23, 42), -1)
    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), primary_color, 1)
    cv2.putText(frame, label_str, (x1 + 4, badge_y2 - 6), font, scale, primary_color, thick, cv2.LINE_AA)

    local_plate_bbox = track_dict.get("local_plate_bbox")
    if local_plate_bbox:
        lpx1, lpy1, lpx2, lpy2 = local_plate_bbox
        cv2.rectangle(frame, (x1 + lpx1, y1 + lpy1), (x1 + lpx2, y1 + lpy2), (50, 50, 255), 2)


def _extract_vehicle_candidates(frame, frame_idx, vehicle_results):
    valid_candidates = []
    for box in vehicle_results.boxes:
        class_id = int(box.cls[0])
        if class_id not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        v_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        if v_crop.size == 0:
            continue

        vehicle_type = VEHICLE_CLASSES[class_id]
        track = vehicle_tracker.match_or_create((x1, y1, x2, y2), vehicle_type, frame_idx)

        if track["color"] is None:
            track["color"] = detect_dominant_color(v_crop)

        valid_candidates.append((track, v_crop, (x1, y1, x2, y2)))
    return valid_candidates


def _is_valid_plate_crop(crop):
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 10:
        return False
    aspect = crop.shape[1] / float(crop.shape[0])
    if not (1.5 <= aspect <= 7.0):
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    adaptive_thresh = max(30.0, min(50.0 * (crop.shape[0] * crop.shape[1] / 3000.0), 200.0))
    return sharpness >= adaptive_thresh


def _dispatch_ocr_task(track, model_name, pool, futures_list, crop, vehicle_crop, area, frame_idx, fps):
    eng_data = track["engines"][model_name]
    has_grew = area > eng_data.get("max_plate_area", 0) * 1.1
    is_cooldown = (frame_idx - eng_data.get("last_ocr_frame", -999) < OCR_COOLDOWN_FRAMES) and not has_grew
    if not is_cooldown and (has_grew or eng_data.get("best_score", 0) < 5000):
        eng_data["last_ocr_frame"] = frame_idx
        futures_list.append(pool.submit(_ocr_worker, crop.copy(), vehicle_crop.copy(), model_name, track["track_id"], area, frame_idx, fps))


def _process_plate_crops(candidates, frame_idx, easyocr_pool, pytesseract_pool, pending_easyocr_futures, pending_pytesseract_futures, fps):
    vehicle_crops = [v[1] for v in candidates]
    batch_plate_results = plate_model(vehicle_crops, verbose=False, conf=0.5)

    for (track, v_crop, _), plate_results in zip(candidates, batch_plate_results):
        if len(plate_results.boxes) == 0:
            continue
        best_box = max(plate_results.boxes, key=lambda p: float(p.conf[0]))
        if float(best_box.conf[0]) < 0.5:
            continue

        lx1, ly1, lx2, ly2 = map(int, best_box.xyxy[0])
        plate_area = (lx2 - lx1) * (ly2 - ly1)
        track["local_plate_bbox"] = (lx1, ly1, lx2, ly2)

        px, py = int((lx2 - lx1) * 0.08), int((ly2 - ly1) * 0.15)
        px1, py1 = max(0, lx1 - px), max(0, ly1 - py)
        px2, py2 = min(v_crop.shape[1], lx2 + px), min(v_crop.shape[0], ly2 + py)
        padded_crop = v_crop[py1:py2, px1:px2]

        if _is_valid_plate_crop(padded_crop):
            _dispatch_ocr_task(track, "EasyOCR", easyocr_pool, pending_easyocr_futures, padded_crop, v_crop, plate_area, frame_idx, fps)
            _dispatch_ocr_task(track, "PyTesseract", pytesseract_pool, pending_pytesseract_futures, padded_crop, v_crop, plate_area, frame_idx, fps)


def _render_model_overlays(frame, frame_idx):
    frame_easy, frame_tess = frame.copy(), frame.copy()
    for t in vehicle_tracker.tracks:
        if frame_idx - t["last_seen"] <= 5:
            _draw_overlay(frame_easy, t, model_theme="EasyOCR")
            _draw_overlay(frame_tess, t, model_theme="PyTesseract")
    return frame_easy, frame_tess


def process_batch_dual(batch_frames, frame_indices, easyocr_pool, pytesseract_pool, pending_easyocr_futures, pending_pytesseract_futures, fps=30.0):
    _harvest_ocr_results(pending_easyocr_futures)
    _harvest_ocr_results(pending_pytesseract_futures)

    resized_batch = [cv2.resize(f, (1920, int(f.shape[0] * (1920 / float(f.shape[1]))))) if f.shape[1] > 1920 else f for f in batch_frames]
    batch_vehicle_results = vehicle_model(resized_batch, verbose=False, conf=MIN_VEHICLE_CONFIDENCE)

    processed_easy_frames, processed_tess_frames = [], []
    for frame, frame_idx, vehicle_results in zip(resized_batch, frame_indices, batch_vehicle_results):
        vehicle_tracker.purge_old(frame_idx, frames_per_second=fps)

        valid_vehicles = _extract_vehicle_candidates(frame, frame_idx, vehicle_results)
        if valid_vehicles:
            _process_plate_crops(valid_vehicles, frame_idx, easyocr_pool, pytesseract_pool, pending_easyocr_futures, pending_pytesseract_futures, fps)

        fe, ft = _render_model_overlays(frame, frame_idx)
        processed_easy_frames.append(fe)
        processed_tess_frames.append(ft)

    return processed_easy_frames, processed_tess_frames