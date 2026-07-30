import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from src.color.color_detector import detect_dominant_color
from src.logging.logger import log_detection, flush_all_logs
from src.ocr.plate_ocr import read_plate

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MIN_VEHICLE_CONFIDENCE = 0.4

device = "cuda" if torch.cuda.is_available() else "cpu"
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("models/yolo_plate/best.pt")
vehicle_model.to(device)
plate_model.to(device)


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


def _ocr_worker(plate_crop_image, vehicle_crop_image, ocr_engine_name, img_id, det_id):
    plate_text, ocr_conf, engine, processed_crop = read_plate(plate_crop_image, ocr_engine_name)

    processed_crop_path = f"outputs/plate_crops/Processed/img{img_id}_det{det_id}_{ocr_engine_name.lower()}.jpg"
    if processed_crop is not None and processed_crop.size > 0:
        cv2.imwrite(processed_crop_path, processed_crop)
    else:
        cv2.imwrite(processed_crop_path, plate_crop_image)

    safe_text = (plate_text or "no_read").replace(' ', '_')
    snapshot_path = f"outputs/snapshots/img{img_id}_det{det_id}_{safe_text}.jpg"
    cv2.imwrite(snapshot_path, vehicle_crop_image)

    return {
        "engine": ocr_engine_name,
        "img_id": img_id,
        "det_id": det_id,
        "plate_text": plate_text,
        "conf": ocr_conf,
        "processed_crop_path": processed_crop_path,
        "snapshot_path": snapshot_path
    }


def _draw_overlay(frame, bbox, vehicle_type, color, plate_text, conf, local_plate_bbox, model_theme="EasyOCR"):
    primary_color = (200, 0, 255) if model_theme == "PyTesseract" else (0, 200, 255)
    annotator = Annotator(frame, line_width=2)

    parts = [vehicle_type.capitalize()]
    if color:
        parts.append(color.capitalize())
    if plate_text:
        parts.append(f"• {plate_text} ({int(conf * 100)}%)")

    label_str = " ".join(parts)
    annotator.box_label(bbox, label_str, color=primary_color)

    if local_plate_bbox:
        x1, y1, _, _ = bbox
        lpx1, lpy1, lpx2, lpy2 = local_plate_bbox
        annotator.box_label((x1 + lpx1, y1 + lpy1, x1 + lpx2, y1 + lpy2), label="Plate", color=(50, 50, 255))


def process_bulk_images(image_paths, easyocr_pool, pytesseract_pool):
    """
    Processes a bulk list of static images.
    Returns a list of dicts with detections per image, and saves annotated versions.
    """
    frames = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        # Resize if too large
        if img.shape[1] > 1920:
            scale = 1920 / float(img.shape[1])
            img = cv2.resize(img, (1920, int(img.shape[0] * scale)))
        frames.append(img)

    if not frames:
        return []

    # Create output directories once before processing
    for d in ["outputs/plate_crops/Processed", "outputs/snapshots", "outputs/annotated"]:
        os.makedirs(d, exist_ok=True)
    # Detect Vehicles in batch
    batch_vehicle_results = vehicle_model(frames, verbose=False, conf=MIN_VEHICLE_CONFIDENCE)

    futures = []
    # img_id -> list of detection dicts (one per vehicle)
    detections_by_image = {i: [] for i in range(len(frames))}
    
    det_counter = 1

    for img_id, (frame, vehicle_results) in enumerate(zip(frames, batch_vehicle_results)):
        vehicle_crops = []
        det_infos = []

        # Find vehicles
        for box in vehicle_results.boxes:
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            v_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if v_crop.size == 0:
                continue

            v_type = VEHICLE_CLASSES[class_id]
            v_color = detect_dominant_color(v_crop)

            det_info = {
                "det_id": det_counter,
                "bbox": (x1, y1, x2, y2),
                "vehicle_type": v_type,
                "color": v_color,
                "v_crop": v_crop,
                "local_plate_bbox": None
            }
            det_infos.append(det_info)
            vehicle_crops.append(v_crop)
            det_counter += 1

        if not vehicle_crops:
            continue

        # Detect plates in batch
        batch_plate_results = plate_model(vehicle_crops, verbose=False, conf=0.5)

        for det_info, plate_results in zip(det_infos, batch_plate_results):
            if len(plate_results.boxes) == 0:
                detections_by_image[img_id].append(det_info)
                continue
            
            best_box = max(plate_results.boxes, key=lambda p: float(p.conf[0]))
            if float(best_box.conf[0]) < 0.5:
                detections_by_image[img_id].append(det_info)
                continue

            lx1, ly1, lx2, ly2 = map(int, best_box.xyxy[0])
            det_info["local_plate_bbox"] = (lx1, ly1, lx2, ly2)

            px, py = int((lx2 - lx1) * 0.08), int((ly2 - ly1) * 0.15)
            px1, py1 = max(0, lx1 - px), max(0, ly1 - py)
            px2, py2 = min(det_info["v_crop"].shape[1], lx2 + px), min(det_info["v_crop"].shape[0], ly2 + py)
            padded_crop = det_info["v_crop"][py1:py2, px1:px2]

            if _is_valid_plate_crop(padded_crop):
                futures.append(easyocr_pool.submit(
                    _ocr_worker, padded_crop.copy(), det_info["v_crop"].copy(), "EasyOCR", img_id, det_info["det_id"]
                ))
                futures.append(pytesseract_pool.submit(
                    _ocr_worker, padded_crop.copy(), det_info["v_crop"].copy(), "PyTesseract", img_id, det_info["det_id"]
                ))
            
            detections_by_image[img_id].append(det_info)

    # Wait for all OCR futures
    ocr_results = [f.result() for f in futures]

    # Map OCR results back to detections
    # Det ID -> { "EasyOCR": res, "PyTesseract": res }
    ocr_by_det = {}
    for res in ocr_results:
        did = res["det_id"]
        if did not in ocr_by_det:
            ocr_by_det[did] = {}
        ocr_by_det[did][res["engine"]] = res

    # Annotate and save images, structure final output
    
    final_output = []

    for img_id, frame in enumerate(frames):
        frame_easy = frame.copy()
        frame_tess = frame.copy()

        img_detections = []
        for det_info in detections_by_image[img_id]:
            did = det_info["det_id"]
            easy_res = ocr_by_det.get(did, {}).get("EasyOCR", {})
            tess_res = ocr_by_det.get(did, {}).get("PyTesseract", {})

            # Draw
            _draw_overlay(frame_easy, det_info["bbox"], det_info["vehicle_type"], det_info["color"], 
                          easy_res.get("plate_text"), easy_res.get("conf", 0.0), det_info["local_plate_bbox"], "EasyOCR")
            _draw_overlay(frame_tess, det_info["bbox"], det_info["vehicle_type"], det_info["color"], 
                          tess_res.get("plate_text"), tess_res.get("conf", 0.0), det_info["local_plate_bbox"], "PyTesseract")

            # Log to CSV (optional, matching old behavior)
            if easy_res.get("plate_text"):
                log_detection(did, det_info["vehicle_type"], det_info["color"], easy_res.get("plate_text"), 
                              easy_res.get("conf", 0.0), easy_res.get("snapshot_path"), easy_res.get("processed_crop_path"), log_path="outputs/logs/detections_easyocr.csv")
            if tess_res.get("plate_text"):
                log_detection(did, det_info["vehicle_type"], det_info["color"], tess_res.get("plate_text"), 
                              tess_res.get("conf", 0.0), tess_res.get("snapshot_path"), tess_res.get("processed_crop_path"), log_path="outputs/logs/detections_pytesseract.csv")

            img_detections.append({
                "det_id": did,
                "vehicle_type": det_info["vehicle_type"],
                "color": det_info["color"],
                "easyocr": {
                    "plate_text": easy_res.get("plate_text"),
                    "conf": easy_res.get("conf", 0.0),
                    "snapshot_url": "/" + str(easy_res.get("snapshot_path")).replace("\\", "/") if easy_res.get("snapshot_path") else None,
                    "crop_url": "/" + str(easy_res.get("processed_crop_path")).replace("\\", "/") if easy_res.get("processed_crop_path") else None,
                },
                "pytesseract": {
                    "plate_text": tess_res.get("plate_text"),
                    "conf": tess_res.get("conf", 0.0),
                    "snapshot_url": "/" + str(tess_res.get("snapshot_path")).replace("\\", "/") if tess_res.get("snapshot_path") else None,
                    "crop_url": "/" + str(tess_res.get("processed_crop_path")).replace("\\", "/") if tess_res.get("processed_crop_path") else None,
                }
            })

        out_name = os.path.basename(image_paths[img_id])
        easy_path = f"outputs/annotated/easy_{out_name}"
        tess_path = f"outputs/annotated/tess_{out_name}"
        cv2.imwrite(easy_path, frame_easy)
        cv2.imwrite(tess_path, frame_tess)

        final_output.append({
            "original_filename": out_name,
            "easyocr_annotated_url": "/" + easy_path,
            "pytesseract_annotated_url": "/" + tess_path,
            "detections": img_detections
        })

    # Flush all CSV logs in a single batch write
    flush_all_logs()

    return final_output