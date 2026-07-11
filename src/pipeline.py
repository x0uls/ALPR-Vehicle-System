import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import itertools
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
from ultralytics import YOLO
from src.color.color_detector import detect_dominant_color
from src.logging.logger import init_log, log_detection, flush_log
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

# ─── Async OCR Configuration ─────────────────────────────────────
# EasyOCR is GPU-bound (PyTorch CUDA serializes via CUDA streams) → 2 workers.
# PyTesseract spawns external OS subprocesses (CPU-bound) → scale to vCPU count.
_ocr_pool = None
_ocr_pool_engine = None

def _get_ocr_pool(engine_name="EasyOCR"):
    global _ocr_pool, _ocr_pool_engine
    if _ocr_pool is not None and _ocr_pool_engine == engine_name:
        return _ocr_pool
    if _ocr_pool is not None:
        _ocr_pool.shutdown(wait=False)
    if engine_name == "PyTesseract":
        workers = min(os.cpu_count() or 2, 6)  # Cap at 6 to avoid memory pressure
    else:
        workers = 2
    _ocr_pool = ThreadPoolExecutor(max_workers=workers)
    _ocr_pool_engine = engine_name
    print(f"[Pipeline] OCR thread pool: {workers} workers for {engine_name}")
    return _ocr_pool

# Only ever mutated from the main thread: appended during submission,
# swept during harvest. OCR threads only touch the Future objects
# themselves (which are inherently thread-safe).
_pending_futures = []

OCR_COOLDOWN_FRAMES = 9  # Don't re-OCR the same track within 9 processed frames

init_log()
os.makedirs("outputs/snapshots", exist_ok=True)
os.makedirs("outputs/plate_crops/Raw", exist_ok=True)
os.makedirs("outputs/plate_crops/Processed", exist_ok=True)


class DetectionTracker:
    """
    Academic Rationale (Object Tracking):
    A simple Intersection over Union (IoU) tracker to maintain vehicle identity across consecutive frames.
    Instead of running computationally expensive Re-Identification (ReID) CNNs, IoU mathematically 
    compares the bounding box overlap area between frames. If the overlap ratio exceeds 
    TRACK_IOU_THRESHOLD, it is confidently classified as the same vehicle.
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

    def purge_old(self, frame_idx):
        self.tracks = [t for t in self.tracks if frame_idx - t["last_seen"] <= TRACK_MAX_AGE]

    def match_or_create(self, bbox, vehicle_type, frame_idx):
        # Try to match to existing track
        for track in self.tracks:
            if track["vehicle_type"] == vehicle_type and self._iou(track["bbox"], bbox) >= TRACK_IOU_THRESHOLD:
                track["bbox"] = bbox
                track["last_seen"] = frame_idx
                return track

        # New vehicle
        track = {
            "track_id": next(self._id_counter),
            "bbox": bbox,
            "vehicle_type": vehicle_type,
            "first_seen": frame_idx,
            "last_seen": frame_idx,
            "color": None,
            "best_plate": None,
            "best_score": 0.0,
            "best_ocr_conf": 0.0,
            "max_plate_area": 0,
            "snapshot_path": None,
            "last_ocr_frame": -999,
        }
        self.tracks.append(track)
        return track

    def update_plate(self, track, plate_text, ocr_conf, plate_area, snapshot_path, frame_idx, video_timestamp=""):
        """Update if the new read is better. Score = plate area * OCR confidence.
        This naturally favors closer, larger plates over tiny distant plates with artificially high confidence."""
        score = plate_area * ocr_conf

        # Heuristic 1: Malaysian plates almost always have both letters and numbers.
        # If one is missing, it's likely a partial edge-crop. Penalize heavily.
        has_letters = any(c.isalpha() for c in plate_text)
        has_numbers = any(c.isdigit() for c in plate_text)
        if not (has_letters and has_numbers):
            score *= 0.1
            
        # Heuristic 2: Favor longer reads over short partial reads.
        compact_len = len(plate_text.replace(" ", ""))
        score *= (compact_len / 7.0)

        if score > track.get("best_score", 0):
            track["best_plate"] = plate_text
            track["best_score"] = score
            track["best_ocr_conf"] = ocr_conf
            track["snapshot_path"] = snapshot_path
            
            last_logged_frame = self.global_logged_plates.get(plate_text, -9999)
            if frame_idx - last_logged_frame > 300:
                plate_crop_path = f"outputs/plate_crops/Processed/frame{frame_idx}_track{track['track_id']}_processed.jpg"
                log_detection(
                    track["track_id"],
                    track["vehicle_type"],
                    track["color"],
                    plate_text,
                    ocr_conf,
                    snapshot_path,
                    plate_crop_path=plate_crop_path,
                    video_timestamp=video_timestamp,
                )
                self.global_logged_plates[plate_text] = frame_idx

    def find_by_id(self, track_id):
        """Lookup a track by its ID. Returns None if the track was purged."""
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
    """Runs OCR + disk I/O in a background thread.
    
    All NumPy arrays passed here MUST be .copy()'d by the caller,
    because the main thread's frame buffer is reused/overwritten.
    """
    plate_text, ocr_conf, engine, processed_crop = read_plate(plate_crop, ocr_engine)

    # Save plate crops (disk I/O offloaded from main thread)
    raw_path = f"outputs/plate_crops/Raw/frame{frame_idx}_track{track_id}_raw.jpg"
    proc_path = f"outputs/plate_crops/Processed/frame{frame_idx}_track{track_id}_processed.jpg"
    cv2.imwrite(raw_path, plate_crop)
    if processed_crop is not None:
        cv2.imwrite(proc_path, processed_crop)

    # Save vehicle snapshot if OCR produced a valid read
    snapshot_path = None
    if plate_text and ocr_conf > 0:
        snapshot_path = f"outputs/snapshots/frame{frame_idx}_{plate_text}.jpg"
        cv2.imwrite(snapshot_path, vehicle_crop)

    video_timestamp = format_video_timestamp(frame_idx, fps)
    return plate_text, ocr_conf, track_id, plate_area, frame_idx, snapshot_path, video_timestamp


def _apply_ocr_result(future):
    """Process a single completed OCR future and update the corresponding track."""
    try:
        text, conf, track_id, plate_area, frame_idx_ocr, snapshot_path, video_timestamp = future.result()
        print(f"[OCR Harvest] Track {track_id}: Text='{text}', Conf={conf:.3f}, Snapshot={snapshot_path}")
        if text and conf > 0 and snapshot_path:
            track = detection_tracker.find_by_id(track_id)
            if track:
                track["max_plate_area"] = max(track.get("max_plate_area", 0), plate_area)
                detection_tracker.update_plate(track, text, conf, plate_area, snapshot_path, frame_idx_ocr, video_timestamp=video_timestamp)
    except Exception as e:
        print(f"[OCR] Worker failed for track: {e}")


def _harvest_ocr_results():
    """Non-blocking sweep: collect any finished OCR futures.
    Called at the top of each process_frame() invocation."""
    still_pending = []
    for future in _pending_futures:
        if future.done():
            _apply_ocr_result(future)
        else:
            still_pending.append(future)
    _pending_futures[:] = still_pending


def drain_pending_ocr():
    """Blocking flush — wait for ALL in-flight OCR futures to complete.
    MUST be called after the video loop ends, before flush_all()."""
    for future in _pending_futures:
        _apply_ocr_result(future)
    _pending_futures.clear()
    flush_log()


# ─── Drawing ──────────────────────────────────────────────────────

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


# ─── Main Per-Frame Pipeline ─────────────────────────────────────

def process_frame(frame, frame_idx, ocr_engine="EasyOCR", fps=30.0):
    # ── Phase 0: Harvest any OCR results that completed since last frame ──
    _harvest_ocr_results()

    # ── Early downscale: cap resolution at 1080p ─────────────────
    # 4K frames are 4× more pixels than needed — YOLO resizes to 640px
    # internally anyway. Only downscales; never upscales smaller inputs.
    h, w = frame.shape[:2]
    if w > 1920:
        scale = 1920 / w
        frame = cv2.resize(frame, (1920, int(h * scale)))

    detection_tracker.purge_old(frame_idx)

    # ── Phase 1: Detect vehicles (GPU) ────────────────────────────
    # YOLOv8 CNN for real-time object detection and classification.
    vehicle_results = vehicle_model(frame, verbose=False, conf=MIN_VEHICLE_CONFIDENCE)[0]

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

        # Color detection (KMeans) — only once per track, stays on main thread.
        if track["color"] is None:
            track["color"] = detect_dominant_color(vehicle_crop)
            
        # AGGRESSIVE OPTIMIZATION: If we already have a highly confident, full-length plate,
        # skip plate_model and OCR entirely for this vehicle!
        best_plate = track.get("best_plate") or ""
        best_conf = track.get("best_ocr_conf") or 0.0
        is_confident = (len(best_plate) >= 6 and best_conf > 0.8) or (len(best_plate) >= 4 and best_conf > 0.90)
        if is_confident:
            continue

        valid_vehicles.append((track, vehicle_crop))

    # ── Phase 2: Detect plates — single batched GPU pass ──────────
    # Instead of running the plate CNN sequentially on every crop,
    # we batch all vehicle crops into a single pass for GPU parallelization.
    if valid_vehicles:
        crops = [v[1] for v in valid_vehicles]
        plate_results_batch = plate_model(crops, verbose=False, conf=0.5)

        for (track, vehicle_crop), plate_results in zip(valid_vehicles, plate_results_batch):
            if len(plate_results.boxes) > 0:
                best_plate_box = max(plate_results.boxes, key=lambda p: float(p.conf[0]))

                if float(best_plate_box.conf[0]) >= 0.5:
                    px1, py1, px2, py2 = map(int, best_plate_box.xyxy[0])
                    plate_area = (px2 - px1) * (py2 - py1)

                    # Gate 1: Only OCR if plate is significantly larger or we lack a good read
                    plate_grew = plate_area > track.get("max_plate_area", 0) * 1.1
                    if plate_grew or track.get("best_score", 0) < 5000:

                        # Gate 2: Per-track cooldown — skip if recently OCR'd,
                        # UNLESS the plate grew (closer vehicle = better opportunity)
                        recently_ocrd = frame_idx - track.get("last_ocr_frame", -999) < OCR_COOLDOWN_FRAMES
                        if recently_ocrd and not plate_grew:
                            continue

                        # ── Phase 3: Crop the plate with padding ──
                        pad_x = int((px2 - px1) * 0.08)
                        pad_y = int((py2 - py1) * 0.15)
                        px1_pad, py1_pad = max(0, px1 - pad_x), max(0, py1 - pad_y)
                        px2_pad, py2_pad = min(vehicle_crop.shape[1], px2 + pad_x), min(vehicle_crop.shape[0], py2 + pad_y)
                        plate_crop = vehicle_crop[py1_pad:py2_pad, px1_pad:px2_pad]

                        if plate_crop.size > 0 and plate_crop.shape[0] >= 5 and plate_crop.shape[1] >= 10:
                            # ── Phase 4: Sharpness gate (Laplacian, CPU, fast) ──
                            gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                            variance = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                            
                            if variance < 50.0:
                                continue

                            # ── Phase 5: Submit OCR to thread pool (non-blocking) ──
                            track["last_ocr_frame"] = frame_idx
                            future = _get_ocr_pool(ocr_engine).submit(
                                _ocr_worker,
                                plate_crop.copy(),    # MUST copy — frame buffer is reused by main thread
                                vehicle_crop.copy(),  # MUST copy — same reason
                                ocr_engine,
                                track["track_id"],
                                plate_area,
                                frame_idx,
                                fps,
                            )
                            _pending_futures.append(future)

    # ── Phase 6: Draw overlays using *last known* plate text ──────
    # We draw even if YOLO missed the car for a frame or two (up to 5 frames)
    # to prevent UI flickering. OCR results arrive asynchronously and update
    # the track dict, so overlays naturally show the latest read.
    for track in detection_tracker.tracks:
        if frame_idx - track["last_seen"] <= 5:
            _draw_overlay(frame, track)

    return frame
