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

def _log_track(track_dict, plate_crop_path=None, video_timestamp=None):
    """
    Unified helper function that writes a track's status and telemetry to the logging file.
    """
    log_detection(
        track_dict["track_id"],
        track_dict["vehicle_type"],
        track_dict["color"],
        track_dict["best_plate"],
        track_dict["best_ocr_conf"],
        track_dict["snapshot_path"],
        plate_crop_path=plate_crop_path if plate_crop_path is not None else track_dict.get("plate_crop_path", ""),
        video_timestamp=video_timestamp if video_timestamp is not None else track_dict.get("video_timestamp", "")
    )

# Select GPU accelerator if CUDA is available, otherwise default to CPU execution
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load standard pretrained YOLO models
# yolov8n.pt: Nano model (extremely fast, lightweight vehicle detector)
# models/yolo_plate/best.pt: Custom trained model specifically targeting license plates
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("models/yolo_plate/best.pt")
vehicle_model.to(device)
plate_model.to(device)

# Bounding box overlap threshold (IoU). Bounding boxes in sequential frames overlapping
# by 30% or more are matched to the same vehicle trajectory.
TRACK_IOU_THRESHOLD = 0.3

# Maximum age threshold. Keep tracks in memory for up to 30 frames after they are lost,
# in case the vehicle is temporarily hidden behind a sign or other obstacle.
TRACK_MAX_AGE = 30

# Rate limiting parameter. Skip running OCR on the same track if we already did so
# within the last 9 frames, saving CPU processing power.
OCR_COOLDOWN_FRAMES = 9


class DetectionTracker:
    """
    Maintains vehicle identities and tracks bounding boxes across frames using IoU overlap.
    
    Stores plate prediction state histories and coordinates voter promotions.
    """
    # Monotonically increasing counter to assign unique IDs to each tracked vehicle
    _id_counter = itertools.count(1)

    def __init__(self):
        self.tracks = []
        self.global_logged_plates = {}

    @staticmethod
    def _iou(bounding_box_a, bounding_box_b):
        """
        Calculates Intersection over Union (IoU) overlap ratio between two bounding boxes.
        
        Boxes are represented as [x_min, y_min, x_max, y_max].
        
        --- MATH TUTORIAL FOR YOUR TEACHER ---
        Intersection over Union (IoU) is a standard math trick to check if two bounding boxes
        cover the same object. 
        It divides the overlapping area (Intersection) by the total merged area (Union):
        
        1. Overlap (Intersection): We find where the boxes overlap by comparing their edges:
           - Overlap Width = max(0, min(x2_A, x2_B) - max(x1_A, x1_B))
           - Overlap Height = max(0, min(y2_A, y2_B) - max(y1_A, y1_B))
           - Intersection Area = Overlap Width * Overlap Height
        
        2. Total Combined Space (Union): The sum of both boxes' areas minus the overlap area
           (so we don't count the overlapping part twice):
           - Area of A = Width_A * Height_A
           - Area of B = Width_B * Height_B
           - Union Area = Area of A + Area of B - Intersection Area
        
        3. IoU Ratio = Intersection Area / Union Area
           - A score of 1.0 means they are identical boxes.
           - A score of 0.0 means they don't overlap at all.
        """
        box_a_x1, box_a_y1, box_a_x2, box_a_y2 = bounding_box_a
        box_b_x1, box_b_y1, box_b_x2, box_b_y2 = bounding_box_b
        
        # Calculate coordinates of overlapping region
        inter_x1 = max(box_a_x1, box_b_x1)
        inter_y1 = max(box_a_y1, box_b_y1)
        inter_x2 = min(box_a_x2, box_b_x2)
        inter_y2 = min(box_a_y2, box_b_y2)
        
        # Calculate intersection area
        intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Calculate individual box areas
        area_a = max(0, box_a_x2 - box_a_x1) * max(0, box_a_y2 - box_a_y1)
        area_b = max(0, box_b_x2 - box_b_x1) * max(0, box_b_y2 - box_b_y1)
        
        # Union = Area A + Area B - Intersection Area
        union_area = area_a + area_b - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def purge_old(self, frame_idx, frames_per_second=30.0, force_flush=False):
        """
        Removes vehicle tracks that have left the scene and writes them to the CSV database.
        
        If a vehicle hasn't been seen for TRACK_MAX_AGE frames, it is assumed to have exited.
        """
        active = []
        for track_dict in self.tracks:
            # Check if track has expired or if we are forcing termination at the end of the video
            if force_flush or (frame_idx - track_dict["last_seen"] > TRACK_MAX_AGE):
                # Only log the vehicle if we successfully read a plate for it during its lifetime
                if track_dict["best_plate"] and track_dict["best_score"] > 0:
                    plate_crop_path = track_dict.get("plate_crop_path") or _processed_crop_path(track_dict["last_seen"], track_dict["track_id"])
                    _log_track(track_dict, plate_crop_path=plate_crop_path, video_timestamp=track_dict.get("video_timestamp", format_video_timestamp(track_dict["last_seen"], frames_per_second)))
            else:
                active.append(track_dict)
        self.tracks = active

    def match_or_create(self, bbox, vehicle_type, frame_idx):
        """
        Links a new bounding box detection to an existing vehicle track using IoU overlap.
        
        If no overlapping track exists, initializes and registers a new vehicle track.
        """
        best_iou = 0.0
        best_track = None
        
        # 1. Match by finding the tracking box with the highest IoU overlap
        for track in self.tracks:
            if track["vehicle_type"] == vehicle_type:
                iou = self._iou(track["bbox"], bbox)
                if iou >= TRACK_IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_track = track

        if best_track:
            # Update matching track's position history
            best_track["prev_bbox"] = best_track["bbox"]
            best_track["bbox"] = bbox
            best_track["last_seen"] = frame_idx
            best_track["age"] += 1
            return best_track

        # 2. Create and initialize a new vehicle track dictionary if no match is found
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
            "local_plate_bbox": None,
            "video_timestamp": "",
            "plate_crop_path": "",
            "structural_state": {}
        }
        self.tracks.append(track)
        return track

    def get_max_displacement(self):
        """
        Calculates the maximum horizontal/vertical movement (in pixels) of any active vehicle.
        
        Uses the distance formula to measure motion speed between the current and previous frame.
        
        --- MATH TUTORIAL FOR YOUR TEACHER ---
        To compute motion speed (displacement), we use the Pythagorean Distance Theorem:
        1. Find the center coordinates of the vehicle box in the previous frame (Prev_CX, Prev_CY)
           and the current frame (Curr_CX, Curr_CY).
        2. Calculate the horizontal and vertical difference (distance):
           - dx = Curr_CX - Prev_CX
           - dy = Curr_CY - Prev_CY
        3. The straight-line distance is computed using math.hypot(dx, dy), which is just:
           - distance = sqrt(dx^2 + dy^2)
        """
        displacements = []
        for track in self.tracks:
            if track.get("prev_bbox") is not None:
                curr_x1, curr_y1, curr_x2, curr_y2 = track["bbox"]
                prev_x1, prev_y1, prev_x2, prev_y2 = track["prev_bbox"]
                # Calculate straight-line Euclidean distance between bounding box centers
                displacement = math.hypot((curr_x1 + curr_x2 - prev_x1 - prev_x2) / 2, (curr_y1 + curr_y2 - prev_y1 - prev_y2) / 2)
                displacements.append(displacement)
        return max(displacements) if displacements else 0.0

    def update_plate(self, track_dict, plate_text, ocr_conf, plate_area, snapshot_path, frame_idx, video_timestamp=""):
        """
        Updates the track's license plate string if the new reading is higher quality.
        
        Uses temporal voting across multiple frames to protect against single-frame errors.
        """
        effective_conf = max(ocr_conf, 0.01) if (plate_text and len(plate_text.strip()) > 0) else ocr_conf
        
        # Calculate a quality score (higher area * higher confidence = better quality read)
        score = plate_area * effective_conf

        # Heuristic 1: Malaysian plates contain both letters and numbers.
        # Penalize strings that are only numbers or only letters (likely partial/misread structures)
        has_letters = any(char.isalpha() for char in plate_text)
        has_numbers = any(char.isdigit() for char in plate_text)
        if not (has_letters and has_numbers):
            score *= 0.1  # Reduce score by 90%
            
        # Heuristic 2: Favor longer reads over short fragments.
        # Multiply score by ratio of length to 7 characters
        compact_plate_length = len(plate_text.replace(" ", ""))
        score *= (compact_plate_length / 7.0)

        # ── Temporal Voting ──
        # Store how many times each unique text variant has been read.
        # This keeps a single high-confidence glitch from overwriting a persistent consensus.
        plate_votes = track_dict.setdefault("plate_votes", {})
        compact_plate_text = plate_text.replace(" ", "")
        plate_votes[compact_plate_text] = plate_votes.get(compact_plate_text, 0) + 1

        is_first_plate_read = track_dict.get("best_plate") is None
        current_best_compact_plate = (track_dict.get("best_plate") or "").replace(" ", "")

        # Flags to override standard voting rules
        has_high_confidence_override = (ocr_conf > 0.10 and track_dict.get("best_ocr_conf", 0) <= 0.01)
        has_better_quality_score = score > track_dict.get("best_score", 0)

        # Voting-aware promotion logic rules:
        # 1. First read: always accept
        # 2. Same text: update if score is better
        # 3. New/different text: accept only if voted at least twice, OR if quality score is 2x higher
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
        """Finds and returns an active track dictionary matching the given ID."""
        for track in self.tracks:
            if track["track_id"] == track_id:
                return track
        return None

    def flush_all(self):
        """Clears all active vehicle tracks and cache maps to prepare for the next run."""
        self.tracks.clear()
        self.global_logged_plates.clear()


# Global tracking instance
detection_tracker = DetectionTracker()


def _ocr_worker(plate_crop_image, vehicle_crop_image, ocr_engine_name, track_id, plate_pixel_area, frame_idx, frames_per_second=30.0):
    """
    Worker function executed in background threads to handle OCR processing and disk writes.
    
    Prevents file operations and CPU-bound OCR models from blocking the main video reader loop.
    """
    plate_text, ocr_conf, engine, processed_crop = read_plate(plate_crop_image, ocr_engine_name)

    # Save crop images to the outputs directory
    raw_path = _raw_crop_path(frame_idx, track_id)
    processed_crop_path = _processed_crop_path(frame_idx, track_id)
    cv2.imwrite(raw_path, plate_crop_image)
    if processed_crop is not None:
        cv2.imwrite(processed_crop_path, processed_crop)

    # If we got a valid plate text, save the full vehicle context image
    snapshot_path = None
    if plate_text and ocr_conf > 0:
        snapshot_path = _snapshot_path(frame_idx, plate_text)
        cv2.imwrite(snapshot_path, vehicle_crop_image)

    video_timestamp = format_video_timestamp(frame_idx, frames_per_second)
    return plate_text, ocr_conf, track_id, plate_pixel_area, frame_idx, snapshot_path, video_timestamp

def _apply_ocr_result(ocr_future):
    """
    Applies the result of a completed background OCR task back to its tracking object.
    """
    try:
        text, conf, track_id, plate_pixel_area, frame_idx_ocr, snapshot_path, video_timestamp = ocr_future.result()
        print(f"[OCR Harvest] Track {track_id}: Text='{text}', Conf={conf:.3f}, Snapshot={snapshot_path}")
        if text and conf > 0 and snapshot_path:
            track = detection_tracker.find_by_id(track_id)
            if track:
                track["max_plate_area"] = max(track.get("max_plate_area", 0), plate_pixel_area)
                updated = detection_tracker.update_plate(track, text, conf, plate_pixel_area, snapshot_path, frame_idx_ocr, video_timestamp=video_timestamp)
                
                # Real-time log write so dashboard UI updates immediately
                if updated:
                    _log_track(track)
    except Exception as e:
        print(f"[OCR] Worker failed: {e}")

def _harvest_ocr_results(pending_ocr_futures):
    """
    Checks the status of background OCR threads in a non-blocking way.
    
    Harvests results from tasks that are finished and keeps unresolved ones in the list.
    """
    unresolved_futures = []
    for ocr_future in pending_ocr_futures:
        if ocr_future.done():
            _apply_ocr_result(ocr_future)
        else:
            unresolved_futures.append(ocr_future)
    # Update the thread pool tracking list in-place
    pending_ocr_futures[:] = unresolved_futures

def drain_pending_ocr(pending_ocr_futures=None):
    """
    Blocks execution and waits for all active background OCR threads to complete.
    
    Called when the video ends to make sure no final predictions are dropped.
    """
    futures_list = pending_ocr_futures if pending_ocr_futures is not None else []
    for ocr_future in futures_list:
        _apply_ocr_result(ocr_future)
    futures_list.clear()
    flush_log()


def _draw_overlay(frame, track_dict):
    """
    Draws vehicle tracking bounding box, plate label overlays, and plate boxes.
    """
    x1, y1, x2, y2 = track_dict["bbox"]
    # Draw green vehicle box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    parts = [track_dict["vehicle_type"]]
    if track_dict["color"]:
        parts.append(track_dict["color"])
    if track_dict["best_plate"]:
        parts.append(track_dict["best_plate"])
    # Write details above bounding box
    cv2.putText(frame, " ".join(parts), (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw plate bounding box in red, calculated relative to current vehicle box position
    local_plate_bbox = track_dict.get("local_plate_bbox")
    if local_plate_bbox:
        lpx1, lpy1, lpx2, lpy2 = local_plate_bbox
        global_plate_x1 = x1 + lpx1
        global_plate_y1 = y1 + lpy1
        global_plate_x2 = x1 + lpx2
        global_plate_y2 = y1 + lpy2
        cv2.rectangle(frame, (global_plate_x1, global_plate_y1), (global_plate_x2, global_plate_y2), (0, 0, 255), 2)


def process_batch(batch_frames, frame_indices, ocr_engine_name, ocr_pool, pending_ocr_futures, fps=30.0):
    """
    Runs vehicle and plate detection models on a micro-batch of frames.
    
    Filters candidates, crops region coordinates, runs sharpness tests, and schedules OCR tasks.
    """
    # 1. Harvest finished background OCR predictions
    _harvest_ocr_results(pending_ocr_futures)

    # 2. Downscale frames exceeding 1080p width to optimize detection speed and memory usage
    resized_batch_frames = []
    for frame in batch_frames:
        h, w = frame.shape[:2]
        if w > 1920:
            scale = 1920 / w
            resized_batch_frames.append(cv2.resize(frame, (1920, int(h * scale))))
        else:
            resized_batch_frames.append(frame)

    # 3. Run YOLO vehicle model on all frames in the batch simultaneously (batch inference)
    batch_vehicle_results = vehicle_model(resized_batch_frames, verbose=False, conf=MIN_VEHICLE_CONFIDENCE)

    processed_frames = []

    # 4. Process detection results frame-by-frame
    for idx, (frame, frame_idx, vehicle_results) in enumerate(zip(resized_batch_frames, frame_indices, batch_vehicle_results)):
        # Flush expired tracks
        detection_tracker.purge_old(frame_idx, frames_per_second=fps)

        valid_vehicle_candidates = []
        for box in vehicle_results.boxes:
            class_id = int(box.cls[0])
            # Filter classes: ignore non-vehicle classes (like persons or signs)
            if class_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_crop_image = frame[y1:y2, x1:x2]
            if vehicle_crop_image.size == 0:
                continue

            vehicle_type = VEHICLE_CLASSES[class_id]
            # Link box to tracker history
            track = detection_tracker.match_or_create((x1, y1, x2, y2), vehicle_type, frame_idx)

            # Analyze vehicle color if it hasn't been set yet
            if track["color"] is None:
                track["color"] = detect_dominant_color(vehicle_crop_image)

            # OCR Bypass rule: if plate reading is already long (>=6 chars) and highly confident (>80%),
            # or moderately long (>=4 chars) and extremely confident (>90%), skip OCR to save processing
            best_plate_text = track.get("best_plate") or ""
            best_confidence_score = track.get("best_ocr_conf") or 0.0
            is_plate_reading_confident = (len(best_plate_text) >= 6 and best_confidence_score > 0.8) or (len(best_plate_text) >= 4 and best_confidence_score > 0.90)
            if is_plate_reading_confident:
                continue

            valid_vehicle_candidates.append((track, vehicle_crop_image, (x1, y1, x2, y2)))

        # 5. Run license plate model inside vehicle boxes
        if valid_vehicle_candidates:
            vehicle_crop_images = [v[1] for v in valid_vehicle_candidates]
            # Bounding box confidence threshold: 0.5. Discards low-confidence plate box proposals.
            batch_plate_results = plate_model(vehicle_crop_images, verbose=False, conf=0.5)

            for (track, vehicle_crop_image, vehicle_bbox), plate_results in zip(valid_vehicle_candidates, batch_plate_results):
                if len(plate_results.boxes) > 0:
                    # Select the license plate candidate box with the highest confidence
                    highest_conf_plate_box = max(plate_results.boxes, key=lambda p: float(p.conf[0]))

                    if float(highest_conf_plate_box.conf[0]) >= 0.5:
                        local_plate_x1, local_plate_y1, local_plate_x2, local_plate_y2 = map(int, highest_conf_plate_box.xyxy[0])
                        plate_pixel_area = (local_plate_x2 - local_plate_x1) * (local_plate_y2 - local_plate_y1)

                        # Project plate's local crop coordinates back to global canvas space for rendering
                        vehicle_box_x1, vehicle_box_y1, _, _ = vehicle_bbox
                        global_plate_x1 = vehicle_box_x1 + local_plate_x1
                        global_plate_y1 = vehicle_box_y1 + local_plate_y1
                        global_plate_x2 = vehicle_box_x1 + local_plate_x2
                        global_plate_y2 = vehicle_box_y1 + local_plate_y2
                        track["global_plate_bbox"] = (global_plate_x1, global_plate_y1, global_plate_x2, global_plate_y2)
                        track["local_plate_bbox"] = (local_plate_x1, local_plate_y1, local_plate_x2, local_plate_y2)

                        # Gate 1: Check if plate size grew by 10% or if we lack a solid reading (best_score < 5000).
                        # Closer vehicles have larger plates and yield better quality readings.
                        has_plate_size_increased = plate_pixel_area > track.get("max_plate_area", 0) * 1.1
                        if not has_plate_size_increased and track.get("best_score", 0) >= 5000:
                            continue
                            
                        # Gate 2: Cooldown check. Skip OCR if we already processed it within the cooldown window
                        # unless the plate size grew (closer vehicle).
                        is_recently_processed_ocr = frame_idx - track.get("last_ocr_frame", -999) < OCR_COOLDOWN_FRAMES
                        if is_recently_processed_ocr and not has_plate_size_increased:
                            continue

                        # Crop plate box coordinates with a padding margins to avoid clipping characters.
                        # padding_x=8% of width, padding_y=15% of height is standard.
                        padding_x = int((local_plate_x2 - local_plate_x1) * 0.08)
                        padding_y = int((local_plate_y2 - local_plate_y1) * 0.15)
                        padded_plate_x1, padded_plate_y1 = max(0, local_plate_x1 - padding_x), max(0, local_plate_y1 - padding_y)
                        padded_plate_x2, padded_plate_y2 = min(vehicle_crop_image.shape[1], local_plate_x2 + padding_x), min(vehicle_crop_image.shape[0], local_plate_y2 + padding_y)
                        padded_plate_crop = vehicle_crop_image[padded_plate_y1:padded_plate_y2, padded_plate_x1:padded_plate_x2]

                        # Ignore tiny crops that cannot contain readable characters
                        if padded_plate_crop.size == 0 or padded_plate_crop.shape[0] < 5 or padded_plate_crop.shape[1] < 10:
                            continue

                        # Aspect ratio filter: Malaysian plates are rectangular, typically 3:1 to 5:1 (ratio 1.5 to 7.0)
                        plate_aspect_ratio = padded_plate_crop.shape[1] / padded_plate_crop.shape[0]
                        if not (1.5 <= plate_aspect_ratio <= 7.0):
                            continue

                        # Sharpness filter: calculates the variance of the Laplacian of the image.
                        # High variance means sharp changes (in-focus text edges); low variance means blurry.
                        grayscale_plate_crop = cv2.cvtColor(padded_plate_crop, cv2.COLOR_BGR2GRAY)
                        sharpness_variance = cv2.Laplacian(grayscale_plate_crop, cv2.CV_64F).var()
                        
                        # Sharpness threshold scales dynamically with resolution.
                        # total_plate_pixels / 3000 adapts the threshold to crop size (min threshold 30.0, max 200.0)
                        total_plate_pixels = padded_plate_crop.shape[0] * padded_plate_crop.shape[1]
                        adaptive_sharpness_threshold = max(30.0, min(50.0 * (total_plate_pixels / 3000.0), 200.0))
                        
                        if sharpness_variance < adaptive_sharpness_threshold:
                            continue

                        # Submit the crop to the background worker pool to execute OCR asynchronously
                        track["last_ocr_frame"] = frame_idx
                        ocr_future = ocr_pool.submit(
                            _ocr_worker,
                            padded_plate_crop.copy(),
                            vehicle_crop_image.copy(),
                            ocr_engine_name,
                            track["track_id"],
                            plate_pixel_area,
                            frame_idx,
                            fps
                        )
                        pending_ocr_futures.append(ocr_future)

        # 6. Render bounding box overlays on active tracks seen within the last 5 frames
        for track in detection_tracker.tracks:
            if frame_idx - track["last_seen"] <= 5:
                _draw_overlay(frame, track)

        processed_frames.append(frame)

    return processed_frames
