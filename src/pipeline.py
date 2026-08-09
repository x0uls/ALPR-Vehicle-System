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
MIN_VEHICLE_CONFIDENCE = 0.4

device = "cuda" if torch.cuda.is_available() else "cpu"
vehicle_model = YOLO("yolov8n.pt").to(device)
plate_model_path = "models/yolo_plate/best.pt"
plate_model = YOLO(plate_model_path).to(device) if os.path.exists(plate_model_path) else None


def _to_base64_url(img):
    if img is None or img.size == 0:
        return ""
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def _ocr_worker(plate_crop, vehicle_crop, engine_name, img_id, det_id):
    plate_text, ocr_conf, engine, proc_crop, latency_ms = read_plate(plate_crop, engine_name)

    proc_b64 = _to_base64_url(proc_crop if proc_crop is not None and proc_crop.size > 0 else plate_crop)
    snap_b64 = _to_base64_url(vehicle_crop)

    return {
        "engine": engine_name, "det_id": det_id, "plate_text": plate_text,
        "conf": ocr_conf, "latency_ms": latency_ms,
        "processed_crop_path": proc_b64, "snapshot_path": snap_b64
    }


def _draw_overlay(frame, bbox, v_type, text, conf, local_plate_bbox, theme):
    color_rgb = (200, 0, 255) if theme == "PyTesseract" else (0, 200, 255)
    ann = Annotator(frame, line_width=2)
    label = f"{v_type.capitalize()} • {text} ({int(conf * 100)}%)" if text else f"{v_type.capitalize()}"
    ann.box_label(bbox, label, color=color_rgb)
    if local_plate_bbox:
        x1, y1, _, _ = bbox
        lx1, ly1, lx2, ly2 = local_plate_bbox
        ann.box_label((x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2), label="Plate", color=(50, 50, 255))


def _decode_image(item):
    if isinstance(item, str):
        return os.path.basename(item), cv2.imread(item)
    elif isinstance(item, tuple):
        fname, img_bytes = item
        if isinstance(img_bytes, bytes):
            nparr = np.frombuffer(img_bytes, np.uint8)
            return fname, cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return None, None


def process_bulk_images(images_input, easyocr_pool, pytesseract_pool):
    frames, valid_filenames = [], []

    for item in images_input:
        fname, img = _decode_image(item)
        if img is None:
            continue
        if img.shape[1] > 1920:
            img = cv2.resize(img, (1920, int(img.shape[0] * (1920 / float(img.shape[1])))))
        frames.append(img)
        valid_filenames.append(fname)

    if not frames:
        return {"results": [], "discarded_stats": {"total_discarded": 0, "no_vehicle_count": 0, "no_plate_count": 0, "discarded_files": []}}

    batch_v_res = vehicle_model(frames, verbose=False, conf=MIN_VEHICLE_CONFIDENCE)
    futures, detections_by_image, discarded_files = [], {i: [] for i in range(len(frames))}, []
    det_counter = 1

    for img_id, (frame, v_res) in enumerate(zip(frames, batch_v_res)):
        out_name = valid_filenames[img_id]
        candidates = []
        for box in v_res.boxes:
            cid = int(box.cls[0])
            if cid not in VEHICLE_CLASSES: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            v_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if v_crop.size > 0:
                candidates.append({"bbox": (x1, y1, x2, y2), "area": (x2-x1)*(y2-y1), "v_type": VEHICLE_CLASSES[cid], "v_crop": v_crop})

        if not candidates:
            discarded_files.append({"filename": out_name, "reason": "No car detected"})
            continue

        main_v = max(candidates, key=lambda c: c["area"])
        v_crop, (vh, vw) = main_v["v_crop"], main_v["v_crop"].shape[:2]
        det_info = {"det_id": det_counter, "bbox": main_v["bbox"], "v_type": main_v["v_type"], "v_crop": v_crop, "local_plate_bbox": None}
        padded_crop = None

        if plate_model:
            p_res = plate_model(v_crop, verbose=False, conf=0.04)
            valid_p = []
            if len(p_res) > 0 and len(p_res[0].boxes) > 0:
                for pbox in p_res[0].boxes:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                    p_conf, pw, ph = float(pbox.conf[0]), px2 - px1, py2 - py1
                    if (py1 + py2)/2.0 >= 0.15 * vh and pw >= 30 and ph >= 10 and 1.5 <= (pw / float(ph) if ph else 0) <= 7.0:
                        valid_p.append((px1, py1, px2, py2, p_conf))

            if valid_p:
                lx1, ly1, lx2, ly2, _ = max(valid_p, key=lambda p: p[4] * (p[2]-p[0]) * (p[3]-p[1]))
                det_info["local_plate_bbox"] = (lx1, ly1, lx2, ly2)
                # Tight crop with 2px padding margin to isolate exact license plate without car body panels
                pad_m = 2
                padded_crop = v_crop[max(0, ly1-pad_m):min(vh, ly2+pad_m), max(0, lx1-pad_m):min(vw, lx2+pad_m)]

        if padded_crop is None or padded_crop.size == 0:
            discarded_files.append({"filename": out_name, "reason": "No plate detected"})
            continue

        det_counter += 1
        futures.append(easyocr_pool.submit(_ocr_worker, padded_crop.copy(), v_crop.copy(), "EasyOCR", img_id, det_info["det_id"]))
        futures.append(pytesseract_pool.submit(_ocr_worker, padded_crop.copy(), v_crop.copy(), "PyTesseract", img_id, det_info["det_id"]))
        detections_by_image[img_id].append(det_info)

    ocr_by_det = {}
    for res in [f.result() for f in futures]:
        ocr_by_det.setdefault(res["det_id"], {})[res["engine"]] = res

    final_output = []
    for img_id, frame in enumerate(frames):
        if not detections_by_image[img_id]: continue
        frame_easy, frame_tess, img_detections = frame.copy(), frame.copy(), []

        for d_info in detections_by_image[img_id]:
            did = d_info["det_id"]
            e_res = ocr_by_det.get(did, {}).get("EasyOCR", {})
            t_res = ocr_by_det.get(did, {}).get("PyTesseract", {})

            _draw_overlay(frame_easy, d_info["bbox"], d_info["v_type"], e_res.get("plate_text"), e_res.get("conf", 0.0), d_info["local_plate_bbox"], "EasyOCR")
            _draw_overlay(frame_tess, d_info["bbox"], d_info["v_type"], t_res.get("plate_text"), t_res.get("conf", 0.0), d_info["local_plate_bbox"], "PyTesseract")

            img_detections.append({
                "det_id": did, "vehicle_type": d_info["v_type"],
                "easyocr": {"plate_text": e_res.get("plate_text"), "conf": e_res.get("conf", 0.0), "latency_ms": e_res.get("latency_ms", 0.0), "snapshot_url": e_res.get("snapshot_path"), "crop_url": e_res.get("processed_crop_path")},
                "pytesseract": {"plate_text": t_res.get("plate_text"), "conf": t_res.get("conf", 0.0), "latency_ms": t_res.get("latency_ms", 0.0), "snapshot_url": t_res.get("snapshot_path"), "crop_url": t_res.get("processed_crop_path")}
            })

        out_name = valid_filenames[img_id]
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