import os
import base64
os.environ["OMP_THREAD_LIMIT"] = "1"
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from src.ocr.plate_ocr import read_plate

# COCO vehicle classes: Class 2 ("car") automatically detects sedans, hatchbacks, SUVs, MPVs, and vans
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# Minimum confidence threshold for YOLO vehicle detections — anything below 0.4 is ignored
MIN_VEHICLE_CONFIDENCE = 0.4

# Load YOLO models once at import time (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
vehicle_model = YOLO("yolov8n.pt").to(device)
plate_model_path = "models/yolo_plate/best.pt"
plate_model = YOLO(plate_model_path).to(device) if os.path.exists(plate_model_path) else None


def _to_base64_url(img):
    """Converts an OpenCV image (numpy array) into a base64-encoded JPEG data URL string.
    This lets us embed images directly in the JSON response without saving files to disk."""
    if img is None or img.size == 0:
        return ""
    # Encode the numpy array as a JPEG with 85% quality
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return ""
    # Convert raw bytes to a base64 string, then wrap it in a data URL
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def _ocr_worker(plate_crop, vehicle_crop, engine_name, img_id, det_id):
    """Worker function that runs in a thread pool. Takes a plate crop and runs OCR on it
    using either EasyOCR or PyTesseract (determined by engine_name).
    Returns a dict with the OCR text, confidence, latency, and base64 images."""

    # read_plate() handles all preprocessing, OCR inference, and post-processing
    plate_text, ocr_conf, engine, proc_crop, latency_ms = read_plate(plate_crop, engine_name)

    # Convert the processed plate crop and the vehicle snapshot to base64 for the frontend
    proc_b64 = _to_base64_url(proc_crop if proc_crop is not None and proc_crop.size > 0 else plate_crop)
    snap_b64 = _to_base64_url(vehicle_crop)

    return {
        "engine": engine_name, "det_id": det_id, "plate_text": plate_text,
        "conf": ocr_conf, "latency_ms": latency_ms,
        "processed_crop_path": proc_b64, "snapshot_path": snap_b64
    }


def _draw_overlay(frame, bbox, v_type, text, conf, local_plate_bbox, theme):
    """Draws bounding boxes and labels on a frame for one detection.
    Uses different colors per OCR engine: magenta for PyTesseract, cyan for EasyOCR."""
    color_rgb = (200, 0, 255) if theme == "PyTesseract" else (0, 200, 255)
    ann = Annotator(frame, line_width=2)
    # Build label text: "Car • ABC 1234 (85%)" or just "Car" if no plate text
    label = f"{v_type.capitalize()} • {text} ({int(conf * 100)}%)" if text else f"{v_type.capitalize()}"
    # Draw the vehicle bounding box with the label
    ann.box_label(bbox, label, color=color_rgb)
    # If a plate bounding box was detected, draw it too (coordinates are relative to vehicle crop,
    # so we offset them by the vehicle bbox's top-left corner)
    if local_plate_bbox:
        x1, y1, _, _ = bbox
        lx1, ly1, lx2, ly2 = local_plate_bbox
        ann.box_label((x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2), label="Plate", color=(50, 50, 255))


def _decode_image(item):
    """Converts a pipeline input item into (filename, opencv_image).
    Accepts either a file path string or a (filename, raw_bytes) tuple from the upload."""
    if isinstance(item, str):
        # File path — read directly from disk
        return os.path.basename(item), cv2.imread(item)
    elif isinstance(item, tuple):
        # (filename, bytes) tuple from the uploaded FormData
        fname, img_bytes = item
        if isinstance(img_bytes, bytes):
            # Decode raw bytes into a numpy array, then into an OpenCV BGR image
            nparr = np.frombuffer(img_bytes, np.uint8)
            return fname, cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return None, None


def process_bulk_images(images_input, easyocr_pool, pytesseract_pool):
    """Main processing pipeline. Takes a list of image inputs, runs vehicle detection,
    plate detection, dual OCR, and returns annotated results with base64 images.

    Args:
        images_input: List of (filename, raw_bytes) tuples from the upload
        easyocr_pool: ThreadPoolExecutor for EasyOCR workers (2 threads)
        pytesseract_pool: ThreadPoolExecutor for PyTesseract workers (up to 6 threads)

    Returns:
        Dict with "results" (per-image detection data) and "discarded_stats"
    """
    frames, valid_filenames = [], []

    # ── Step 1: Decode all uploaded images into OpenCV arrays ──
    for item in images_input:
        fname, img = _decode_image(item)
        if img is None:
            continue
        # Cap image width at 1920px to keep processing fast
        if img.shape[1] > 1920:
            img = cv2.resize(img, (1920, int(img.shape[0] * (1920 / float(img.shape[1])))))
        frames.append(img)
        valid_filenames.append(fname)

    # Nothing to process — return empty results
    if not frames:
        return {"results": [], "discarded_stats": {"total_discarded": 0, "no_vehicle_count": 0, "no_plate_count": 0, "discarded_files": []}}

    # ── Step 2: Batch vehicle detection with YOLOv8n ──
    # Runs inference on ALL frames in one call for efficiency (GPU batching)
    batch_v_res = vehicle_model(frames, verbose=False, conf=MIN_VEHICLE_CONFIDENCE)
    # futures: collects thread pool futures for OCR tasks
    # detections_by_image: maps image index → list of detection info dicts
    # discarded_files: tracks images that got filtered out and why
    futures, detections_by_image, discarded_files = [], {i: [] for i in range(len(frames))}, []
    det_counter = 1  # Global detection ID counter across all images

    # ── Step 3: Process each image's YOLO results ──
    for img_id, (frame, v_res) in enumerate(zip(frames, batch_v_res)):
        out_name = valid_filenames[img_id]
        candidates = []

        # Loop through every bounding box YOLO detected in this frame
        for box in v_res.boxes:
            # box.cls is a tensor like tensor([2.]) — extract the class ID as an integer
            cid = int(box.cls[0])
            # Skip non-vehicle detections (people, animals, etc.)
            if cid not in VEHICLE_CLASSES: continue
            # box.xyxy[0] is a tensor of [x1, y1, x2, y2] — convert to ints
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Crop the vehicle region from the frame (clamped to image bounds)
            v_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if v_crop.size > 0:
                candidates.append({"bbox": (x1, y1, x2, y2), "area": (x2-x1)*(y2-y1), "v_type": VEHICLE_CLASSES[cid], "v_crop": v_crop})

        # No vehicles found in this image — discard it
        if not candidates:
            discarded_files.append({"filename": out_name, "reason": "No car detected"})
            continue

        # Pick the biggest vehicle (by bounding box area) as the main detection
        main_v = max(candidates, key=lambda c: c["area"])
        v_crop, (vh, vw) = main_v["v_crop"], main_v["v_crop"].shape[:2]
        det_info = {"det_id": det_counter, "bbox": main_v["bbox"], "v_type": main_v["v_type"], "v_crop": v_crop, "local_plate_bbox": None}
        padded_crop = None

        # ── Step 4: Plate detection within the vehicle crop ──
        if plate_model:
            # Run the custom plate YOLO model on the vehicle crop (low conf=0.04 to catch faint plates)
            p_res = plate_model(v_crop, verbose=False, conf=0.04)
            valid_p = []
            if len(p_res) > 0 and len(p_res[0].boxes) > 0:
                for pbox in p_res[0].boxes:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                    p_conf, pw, ph = float(pbox.conf[0]), px2 - px1, py2 - py1
                    # Filter plate candidates:
                    # - Must be in bottom 85% of vehicle (not on the roof)
                    # - Minimum size 30x10 pixels
                    # - Aspect ratio between 1.5 and 7.0 (plates are wide, not square)
                    if (py1 + py2)/2.0 >= 0.15 * vh and pw >= 30 and ph >= 10 and 1.5 <= (pw / float(ph) if ph else 0) <= 7.0:
                        valid_p.append((px1, py1, px2, py2, p_conf))

            if valid_p:
                # Pick the best plate by confidence × area (balances clarity and size)
                lx1, ly1, lx2, ly2, _ = max(valid_p, key=lambda p: p[4] * (p[2]-p[0]) * (p[3]-p[1]))
                det_info["local_plate_bbox"] = (lx1, ly1, lx2, ly2)
                # Tight crop with 2px padding margin to isolate exact license plate without car body panels
                pad_m = 2
                padded_crop = v_crop[max(0, ly1-pad_m):min(vh, ly2+pad_m), max(0, lx1-pad_m):min(vw, lx2+pad_m)]

        # No valid plate detected — discard this image
        if padded_crop is None or padded_crop.size == 0:
            discarded_files.append({"filename": out_name, "reason": "No plate detected"})
            continue

        # ── Step 5: Submit plate crop to both OCR engines in parallel ──
        det_counter += 1
        # .copy() ensures each thread gets its own numpy array (thread safety)
        futures.append(easyocr_pool.submit(_ocr_worker, padded_crop.copy(), v_crop.copy(), "EasyOCR", img_id, det_info["det_id"]))
        futures.append(pytesseract_pool.submit(_ocr_worker, padded_crop.copy(), v_crop.copy(), "PyTesseract", img_id, det_info["det_id"]))
        detections_by_image[img_id].append(det_info)

    # ── Step 6: Collect all OCR results from the thread pools ──
    # Wait for all futures to complete, then group results by detection ID and engine name
    ocr_by_det = {}
    for res in [f.result() for f in futures]:
        ocr_by_det.setdefault(res["det_id"], {})[res["engine"]] = res

    # ── Step 7: Assemble final output with annotated images ──
    final_output = []
    for img_id, frame in enumerate(frames):
        # Skip images that had no detections (they were discarded earlier)
        if not detections_by_image[img_id]: continue
        # Make two separate copies of the frame — one for each OCR engine's overlay
        frame_easy, frame_tess, img_detections = frame.copy(), frame.copy(), []

        for d_info in detections_by_image[img_id]:
            did = d_info["det_id"]
            e_res = ocr_by_det.get(did, {}).get("EasyOCR", {})
            t_res = ocr_by_det.get(did, {}).get("PyTesseract", {})

            # Draw bounding boxes and plate text on each frame copy
            _draw_overlay(frame_easy, d_info["bbox"], d_info["v_type"], e_res.get("plate_text"), e_res.get("conf", 0.0), d_info["local_plate_bbox"], "EasyOCR")
            _draw_overlay(frame_tess, d_info["bbox"], d_info["v_type"], t_res.get("plate_text"), t_res.get("conf", 0.0), d_info["local_plate_bbox"], "PyTesseract")

            # Bundle both engines' results into a single detection record
            img_detections.append({
                "det_id": did, "vehicle_type": d_info["v_type"],
                "easyocr": {"plate_text": e_res.get("plate_text"), "conf": e_res.get("conf", 0.0), "latency_ms": e_res.get("latency_ms", 0.0), "snapshot_url": e_res.get("snapshot_path"), "crop_url": e_res.get("processed_crop_path")},
                "pytesseract": {"plate_text": t_res.get("plate_text"), "conf": t_res.get("conf", 0.0), "latency_ms": t_res.get("latency_ms", 0.0), "snapshot_url": t_res.get("snapshot_path"), "crop_url": t_res.get("processed_crop_path")}
            })

        out_name = valid_filenames[img_id]
        # Encode the annotated frames to base64 so the frontend can display them directly
        easy_b64 = _to_base64_url(frame_easy)
        tess_b64 = _to_base64_url(frame_tess)
        final_output.append({
            "original_filename": out_name,
            "easyocr_annotated_url": easy_b64,
            "pytesseract_annotated_url": tess_b64,
            "detections": img_detections
        })

    return {
        "results": final_output,
        "discarded_stats": {
            "total_discarded": len(discarded_files),
            "no_vehicle_count": sum(1 for d in discarded_files if d["reason"] == "No car detected"),
            "no_plate_count": sum(1 for d in discarded_files if d["reason"] == "No plate detected"),
            "discarded_files": discarded_files
        }
    }