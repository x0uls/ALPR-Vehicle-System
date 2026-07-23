import os
# Limit OpenMP threads to 1 to prevent CPU core resource contention during batch model calls
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

# Class index mappings returned by YOLOv8 COCO models:
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Bounding box candidate threshold. Bounding boxes with confidence below 40%
# are discarded early to filter out background noise/false positives.
MIN_VEHICLE_CONFIDENCE = 0.4

# Frame skipping parameter: processes 1 frame out of every 3 frames in heavy modes
# to speed up analysis.
FRAME_SKIP = 3

def format_video_timestamp(frame_idx, frames_per_second):
    """
    Converts a numerical frame index and FPS rate into a standard MM:SS.s timecode string.
    
    This matches frame sequences with real video time.
    """
    if frames_per_second <= 0:
        return "00:00.0"
    minutes, seconds = divmod(frame_idx / frames_per_second, 60)
    return f"{int(minutes):02d}:{seconds:04.1f}"

def _raw_crop_path(frame_idx, track_id):
    """Generates the file path for storing raw license plate crops."""
    return f"outputs/plate_crops/Raw/frame{frame_idx}_track{track_id}_raw.jpg"

def _processed_crop_path(frame_idx, track_id):
    """Generates the file path for storing preprocessed (binarized/deskewed) plate crops."""
    return f"outputs/plate_crops/Processed/frame{frame_idx}_track{track_id}_processed.jpg"

def _snapshot_path(frame_idx, plate_text):
    """Generates the file path for saving full-context vehicle snapshots."""
    return f"outputs/snapshots/frame{frame_idx}_{plate_text.replace(' ', '_')}.jpg"

def _log_track(track_dict, plate_crop_path=None, video_timestamp=None, log_path="outputs/logs/detections.csv"):
    """
    Unified helper function that writes a track's status and telemetry to the specified logging file.
    """
    log_detection(
        track_dict["track_id"],
        track_dict["vehicle_type"],
        track_dict["color"],
        track_dict["best_plate"],
        track_dict["best_ocr_conf"],
        track_dict["snapshot_path"],
        plate_crop_path=plate_crop_path if plate_crop_path is not None else track_dict.get("plate_crop_path", ""),
        video_timestamp=video_timestamp if video_timestamp is not None else track_dict.get("video_timestamp", ""),
        log_path=log_path
    )

# Select GPU accelerator if CUDA is available, otherwise default to CPU execution
device = "cuda" if torch.cuda.is_available() else "cpu"

vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("models/yolo_plate/best.pt")
vehicle_model.to(device)
plate_model.to(device)

TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_AGE = 30
OCR_COOLDOWN_FRAMES = 9


class DetectionTracker:
    """
    Maintains vehicle identities and tracks bounding boxes across frames using IoU overlap.
    """
    def __init__(self, log_path="outputs/logs/detections.csv"):
        self._id_counter = itertools.count(1)
        self.tracks = []
        self.global_logged_plates = {}
        self.log_path = log_path

    @staticmethod
    def _iou(bounding_box_a, bounding_box_b):
        box_a_x1, box_a_y1, box_a_x2, box_a_y2 = bounding_box_a
        box_b_x1, box_b_y1, box_b_x2, box_b_y2 = bounding_box_b
        
        inter_x1 = max(box_a_x1, box_b_x1)
        inter_y1 = max(box_a_y1, box_b_y1)
        inter_x2 = min(box_a_x2, box_b_x2)
        inter_y2 = min(box_a_y2, box_b_y2)
        
        intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = max(0, box_a_x2 - box_a_x1) * max(0, box_a_y2 - box_a_y1)
        area_b = max(0, box_b_x2 - box_b_x1) * max(0, box_b_y2 - box_b_y1)
        union_area = area_a + area_b - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def purge_old(self, frame_idx, frames_per_second=30.0, force_flush=False):
        active = []
        for track_dict in self.tracks:
            if force_flush or (frame_idx - track_dict["last_seen"] > TRACK_MAX_AGE):
                if track_dict["best_plate"] and track_dict["best_score"] > 0:
                    plate_crop_path = track_dict.get("plate_crop_path") or _processed_crop_path(track_dict["last_seen"], track_dict["track_id"])
                    _log_track(
                        track_dict,
                        plate_crop_path=plate_crop_path,
                        video_timestamp=track_dict.get("video_timestamp", format_video_timestamp(track_dict["last_seen"], frames_per_second)),
                        log_path=self.log_path
                    )
            else:
                active.append(track_dict)
        self.tracks = active

    def match_or_create(self, bbox, vehicle_type, frame_idx, override_id=None):
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
            if override_id is not None:
                best_track["track_id"] = override_id
            return best_track

        t_id = override_id if override_id is not None else next(self._id_counter)
        track = {
            "track_id": t_id,
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
            "local_plate_bbox": None,
            "video_timestamp": "",
            "plate_crop_path": "",
            "structural_state": {}
        }
        self.tracks.append(track)
        return track

    def get_max_displacement(self):
        displacements = []
        for track in self.tracks:
            if track.get("prev_bbox") is not None:
                curr_x1, curr_y1, curr_x2, curr_y2 = track["bbox"]
                prev_x1, prev_y1, prev_x2, prev_y2 = track["prev_bbox"]
                displacement = math.hypot((curr_x1 + curr_x2 - prev_x1 - prev_x2) / 2, (curr_y1 + curr_y2 - prev_y1 - prev_y2) / 2)
                displacements.append(displacement)
        return max(displacements) if displacements else 0.0

    def update_plate(self, track_dict, plate_text, ocr_conf, plate_area, snapshot_path, frame_idx, video_timestamp=""):
        effective_conf = max(ocr_conf, 0.01) if (plate_text and len(plate_text.strip()) > 0) else ocr_conf
        score = plate_area * effective_conf

        has_letters = any(char.isalpha() for char in plate_text)
        has_numbers = any(char.isdigit() for char in plate_text)
        if not (has_letters and has_numbers):
            score *= 0.1
            
        compact_plate_length = len(plate_text.replace(" ", ""))
        score *= (compact_plate_length / 7.0)

        plate_votes = track_dict.setdefault("plate_votes", {})
        compact_plate_text = plate_text.replace(" ", "")
        plate_votes[compact_plate_text] = plate_votes.get(compact_plate_text, 0) + 1

        is_first_plate_read = track_dict.get("best_plate") is None
        current_best_compact_plate = (track_dict.get("best_plate") or "").replace(" ", "")

        has_high_confidence_override = (ocr_conf > 0.10 and track_dict.get("best_ocr_conf", 0) <= 0.01)
        has_better_quality_score = score > track_dict.get("best_score", 0)

        if is_first_plate_read or has_high_confidence_override:
            should_update_best_plate = True
        elif compact_plate_text == current_best_compact_plate:
            should_update_best_plate = has_better_quality_score
        else:
            plate_vote_count = plate_votes.get(compact_plate_text, 0)
            should_update_best_plate = (plate_vote_count >= 2 and has_better_quality_score) or (score > track_dict.get("best_score", 0) * 2.0)

        if should_update_best_plate:
            track_dict["best_plate"] = plate_text
            track_dict["best_score"] = max(score, track_dict.get("best_score", 0)) 
            track_dict["best_ocr_conf"] = ocr_conf
            track_dict["snapshot_path"] = snapshot_path
            track_dict["video_timestamp"] = video_timestamp
            track_dict["plate_crop_path"] = _processed_crop_path(frame_idx, track_dict["track_id"])
        return should_update_best_plate

    def find_by_id(self, track_id):
        for track in self.tracks:
            if track["track_id"] == track_id:
                return track
        return None

    def flush_all(self):
        self.tracks.clear()
        self.global_logged_plates.clear()


# Default global tracking instances
easyocr_tracker = DetectionTracker(log_path="outputs/logs/detections_easyocr.csv")
pytesseract_tracker = DetectionTracker(log_path="outputs/logs/detections_pytesseract.csv")
detection_tracker = easyocr_tracker


def _ocr_worker(plate_crop_image, vehicle_crop_image, ocr_engine_name, track_id, plate_pixel_area, frame_idx, frames_per_second=30.0):
    """
    Worker function executed in background threads to handle OCR processing and disk writes.
    """
    plate_text, ocr_conf, engine, processed_crop = read_plate(plate_crop_image, ocr_engine_name)

    raw_path = _raw_crop_path(frame_idx, track_id)
    processed_crop_path = _processed_crop_path(frame_idx, track_id)
    cv2.imwrite(raw_path, plate_crop_image)
    if processed_crop is not None:
        cv2.imwrite(processed_crop_path, processed_crop)

    snapshot_path = None
    if plate_text and ocr_conf > 0:
        snapshot_path = _snapshot_path(frame_idx, plate_text)
        cv2.imwrite(snapshot_path, vehicle_crop_image)

    video_timestamp = format_video_timestamp(frame_idx, frames_per_second)
    return ocr_engine_name, plate_text, ocr_conf, track_id, plate_pixel_area, frame_idx, snapshot_path, video_timestamp, processed_crop_path


def _apply_ocr_result(ocr_future, target_tracker=None):
    """
    Applies the result of a completed background OCR task back to its tracking object.
    """
    try:
        engine_name, text, conf, track_id, plate_pixel_area, frame_idx_ocr, snapshot_path, video_timestamp, crop_path = ocr_future.result()
        tracker = target_tracker if target_tracker is not None else (easyocr_tracker if engine_name == "EasyOCR" else pytesseract_tracker)
        log_path = tracker.log_path
        
        if text and conf > 0 and snapshot_path:
            track = tracker.find_by_id(track_id)
            if track:
                track["max_plate_area"] = max(track.get("max_plate_area", 0), plate_pixel_area)
                updated = tracker.update_plate(track, text, conf, plate_pixel_area, snapshot_path, frame_idx_ocr, video_timestamp=video_timestamp)
                if updated:
                    _log_track(track, plate_crop_path=crop_path, video_timestamp=video_timestamp, log_path=log_path)
    except Exception as e:
        print(f"[OCR] Worker failed: {e}")


def _harvest_ocr_results(pending_ocr_futures, target_tracker=None):
    unresolved_futures = []
    for ocr_future in pending_ocr_futures:
        if ocr_future.done():
            _apply_ocr_result(ocr_future, target_tracker=target_tracker)
        else:
            unresolved_futures.append(ocr_future)
    pending_ocr_futures[:] = unresolved_futures


def drain_pending_ocr(pending_ocr_futures=None, target_tracker=None):
    futures_list = pending_ocr_futures if pending_ocr_futures is not None else []
    for ocr_future in futures_list:
        _apply_ocr_result(ocr_future, target_tracker=target_tracker)
    futures_list.clear()
    target_path = target_tracker.log_path if target_tracker else "outputs/logs/detections.csv"
    flush_log(log_path=target_path)


def _draw_overlay(frame, track_dict, model_theme="EasyOCR"):
    """
    Draws lively, high-tech vehicle tracking bounding box, plate label overlays, and plate boxes.
    """
    x1, y1, x2, y2 = track_dict["bbox"]
    w, h = x2 - x1, y2 - y1
    
    # Select accent colors based on model theme (BGR format)
    if model_theme == "PyTesseract":
        primary_color = (255, 0, 200)   # Vibrant Neon Purple
    else:
        primary_color = (255, 200, 0)   # Bright Azure / Cyan

    # 1. Semi-transparent background box highlight (12% alpha)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), primary_color, -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # 2. Main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), primary_color, 2)

    # 3. High-tech L-shaped corner brackets for a lively tracking effect
    corner_len = min(22, max(8, int(min(w, h) * 0.2)))
    thick = 3
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), primary_color, thick)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), primary_color, thick)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), primary_color, thick)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), primary_color, thick)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), primary_color, thick)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), primary_color, thick)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), primary_color, thick)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), primary_color, thick)

    # 4. Construct live label text
    parts = [f"#{track_dict['track_id']}", track_dict["vehicle_type"].capitalize()]
    if track_dict.get("color"):
        parts.append(track_dict["color"].capitalize())
    
    plate_text = track_dict.get("best_plate")
    conf = track_dict.get("best_ocr_conf", 0.0)
    if plate_text:
        parts.append(f"• {plate_text} ({int(conf * 100)}%)")
    else:
        parts.append("• Scanning...")

    label_str = " " + " ".join(parts) + " "
    
    # 5. Draw dark top badge background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    font_thick = 2
    (text_w, text_h), _ = cv2.getTextSize(label_str, font, font_scale, font_thick)
    
    badge_y1 = max(0, y1 - text_h - 12)
    badge_y2 = y1
    badge_x2 = min(frame.shape[1], x1 + text_w + 10)
    
    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), (15, 23, 42), -1)
    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), primary_color, 1)

    text_color = primary_color
    cv2.putText(frame, label_str, (x1 + 4, badge_y2 - 6), font, font_scale, text_color, font_thick, cv2.LINE_AA)

    # 6. Draw local license plate crop box in vibrant red/coral
    local_plate_bbox = track_dict.get("local_plate_bbox")
    if local_plate_bbox:
        lpx1, lpy1, lpx2, lpy2 = local_plate_bbox
        global_plate_x1 = x1 + lpx1
        global_plate_y1 = y1 + lpy1
        global_plate_x2 = x1 + lpx2
        global_plate_y2 = y1 + lpy2
        cv2.rectangle(frame, (global_plate_x1, global_plate_y1), (global_plate_x2, global_plate_y2), (50, 50, 255), 2)



def process_batch_dual(batch_frames, frame_indices, easyocr_pool, pytesseract_pool, pending_easyocr_futures, pending_pytesseract_futures, fps=30.0):
    """
    Runs vehicle and license plate detection once per frame micro-batch, then passes plate crops
    to BOTH EasyOCR and PyTesseract worker pools simultaneously.
    
    Returns a tuple of (processed_easyocr_frames, processed_pytesseract_frames).
    """
    # 1. Harvest finished background OCR predictions for both models
    _harvest_ocr_results(pending_easyocr_futures, target_tracker=easyocr_tracker)
    _harvest_ocr_results(pending_pytesseract_futures, target_tracker=pytesseract_tracker)

    # 2. Downscale frames exceeding 1080p width
    resized_batch_frames = []
    for frame in batch_frames:
        h, w = frame.shape[:2]
        if w > 1920:
            scale = 1920 / w
            resized_batch_frames.append(cv2.resize(frame, (1920, int(h * scale))))
        else:
            resized_batch_frames.append(frame)

    # 3. Run YOLO vehicle model ONCE on all frames in the batch
    batch_vehicle_results = vehicle_model(resized_batch_frames, verbose=False, conf=MIN_VEHICLE_CONFIDENCE)

    processed_easyocr_frames = []
    processed_pytesseract_frames = []

    # 4. Process frame by frame
    for idx, (frame, frame_idx, vehicle_results) in enumerate(zip(resized_batch_frames, frame_indices, batch_vehicle_results)):
        easyocr_tracker.purge_old(frame_idx, frames_per_second=fps)
        pytesseract_tracker.purge_old(frame_idx, frames_per_second=fps)

        valid_vehicle_candidates = []
        for box in vehicle_results.boxes:
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_crop_image = frame[y1:y2, x1:x2]
            if vehicle_crop_image.size == 0:
                continue

            vehicle_type = VEHICLE_CLASSES[class_id]

            # Match or create tracks on BOTH model trackers with unified track_id
            track_easy = easyocr_tracker.match_or_create((x1, y1, x2, y2), vehicle_type, frame_idx)
            track_tess = pytesseract_tracker.match_or_create((x1, y1, x2, y2), vehicle_type, frame_idx, override_id=track_easy["track_id"])

            # Analyze vehicle color if missing
            if track_easy["color"] is None:
                col = detect_dominant_color(vehicle_crop_image)
                track_easy["color"] = col
                track_tess["color"] = col

            valid_vehicle_candidates.append((track_easy, track_tess, vehicle_crop_image, (x1, y1, x2, y2)))

        # 5. Run license plate model ONCE inside vehicle boxes
        if valid_vehicle_candidates:
            vehicle_crop_images = [v[2] for v in valid_vehicle_candidates]
            batch_plate_results = plate_model(vehicle_crop_images, verbose=False, conf=0.5)

            for (track_easy, track_tess, vehicle_crop_image, vehicle_bbox), plate_results in zip(valid_vehicle_candidates, batch_plate_results):
                if len(plate_results.boxes) > 0:
                    highest_conf_plate_box = max(plate_results.boxes, key=lambda p: float(p.conf[0]))

                    if float(highest_conf_plate_box.conf[0]) >= 0.5:
                        local_plate_x1, local_plate_y1, local_plate_x2, local_plate_y2 = map(int, highest_conf_plate_box.xyxy[0])
                        plate_pixel_area = (local_plate_x2 - local_plate_x1) * (local_plate_y2 - local_plate_y1)

                        vehicle_box_x1, vehicle_box_y1, _, _ = vehicle_bbox
                        global_plate_x1 = vehicle_box_x1 + local_plate_x1
                        global_plate_y1 = vehicle_box_y1 + local_plate_y1
                        global_plate_x2 = vehicle_box_x1 + local_plate_x2
                        global_plate_y2 = vehicle_box_y1 + local_plate_y2

                        track_easy["global_plate_bbox"] = (global_plate_x1, global_plate_y1, global_plate_x2, global_plate_y2)
                        track_easy["local_plate_bbox"] = (local_plate_x1, local_plate_y1, local_plate_x2, local_plate_y2)

                        track_tess["global_plate_bbox"] = (global_plate_x1, global_plate_y1, global_plate_x2, global_plate_y2)
                        track_tess["local_plate_bbox"] = (local_plate_x1, local_plate_y1, local_plate_x2, local_plate_y2)

                        padding_x = int((local_plate_x2 - local_plate_x1) * 0.08)
                        padding_y = int((local_plate_y2 - local_plate_y1) * 0.15)
                        padded_plate_x1, padded_plate_y1 = max(0, local_plate_x1 - padding_x), max(0, local_plate_y1 - padding_y)
                        padded_plate_x2, padded_plate_y2 = min(vehicle_crop_image.shape[1], local_plate_x2 + padding_x), min(vehicle_crop_image.shape[0], local_plate_y2 + padding_y)
                        padded_plate_crop = vehicle_crop_image[padded_plate_y1:padded_plate_y2, padded_plate_x1:padded_plate_x2]

                        if padded_plate_crop.size == 0 or padded_plate_crop.shape[0] < 5 or padded_plate_crop.shape[1] < 10:
                            continue

                        plate_aspect_ratio = padded_plate_crop.shape[1] / padded_plate_crop.shape[0]
                        if not (1.5 <= plate_aspect_ratio <= 7.0):
                            continue

                        grayscale_plate_crop = cv2.cvtColor(padded_plate_crop, cv2.COLOR_BGR2GRAY)
                        sharpness_variance = cv2.Laplacian(grayscale_plate_crop, cv2.CV_64F).var()
                        total_plate_pixels = padded_plate_crop.shape[0] * padded_plate_crop.shape[1]
                        adaptive_sharpness_threshold = max(30.0, min(50.0 * (total_plate_pixels / 3000.0), 200.0))
                        
                        if sharpness_variance < adaptive_sharpness_threshold:
                            continue

                        # Gate 1 & 2 for EasyOCR (check plate growth and cooldown window)
                        has_easy_size_grew = plate_pixel_area > track_easy.get("max_plate_area", 0) * 1.1
                        is_easy_cooldown = (frame_idx - track_easy.get("last_ocr_frame", -999) < OCR_COOLDOWN_FRAMES) and not has_easy_size_grew
                        should_run_easy = not is_easy_cooldown and (has_easy_size_grew or track_easy.get("best_score", 0) < 5000)

                        # Gate 1 & 2 for PyTesseract (check plate growth and cooldown window)
                        has_tess_size_grew = plate_pixel_area > track_tess.get("max_plate_area", 0) * 1.1
                        is_tess_cooldown = (frame_idx - track_tess.get("last_ocr_frame", -999) < OCR_COOLDOWN_FRAMES) and not has_tess_size_grew
                        should_run_tess = not is_tess_cooldown and (has_tess_size_grew or track_tess.get("best_score", 0) < 5000)

                        if should_run_easy:
                            track_easy["last_ocr_frame"] = frame_idx
                            ocr_future_easy = easyocr_pool.submit(
                                _ocr_worker,
                                padded_plate_crop.copy(),
                                vehicle_crop_image.copy(),
                                "EasyOCR",
                                track_easy["track_id"],
                                plate_pixel_area,
                                frame_idx,
                                fps
                            )
                            pending_easyocr_futures.append(ocr_future_easy)

                        if should_run_tess:
                            track_tess["last_ocr_frame"] = frame_idx
                            ocr_future_tess = pytesseract_pool.submit(
                                _ocr_worker,
                                padded_plate_crop.copy(),
                                vehicle_crop_image.copy(),
                                "PyTesseract",
                                track_tess["track_id"],
                                plate_pixel_area,
                                frame_idx,
                                fps
                            )
                            pending_pytesseract_futures.append(ocr_future_tess)

        # 6. Render bounding box overlays on copies of the frame for both models
        frame_easy = frame.copy()
        for track in easyocr_tracker.tracks:
            if frame_idx - track["last_seen"] <= 5:
                _draw_overlay(frame_easy, track, model_theme="EasyOCR")

        frame_tess = frame.copy()
        for track in pytesseract_tracker.tracks:
            if frame_idx - track["last_seen"] <= 5:
                _draw_overlay(frame_tess, track, model_theme="PyTesseract")

        processed_easyocr_frames.append(frame_easy)
        processed_pytesseract_frames.append(frame_tess)

    return processed_easyocr_frames, processed_pytesseract_frames

