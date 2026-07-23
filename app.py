import glob
import os
import time
import subprocess
import asyncio
import base64
import json

import cv2
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import sys

import torch
from src.pipeline import easyocr_tracker, pytesseract_tracker, process_batch_dual, drain_pending_ocr
from src.logging.logger import init_log
from src.metrics.cer import (
    compute_cer, find_best_ground_truth_match,
    save_ground_truth, load_ground_truth, compute_average_cer,
    compute_comprehensive_metrics, compute_dual_model_comparison
)

def _format_elapsed(seconds):
    """
    Formats elapsed raw seconds into a friendly human-readable time string.
    """
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ─── Server-Sent Events (SSE) Real-Time Generator ────────────────

async def process_video_sse(video_path, ocr_engine="dual", frame_skip="dynamic"):
    """
    Asynchronous generator that runs dual-model (EasyOCR + PyTesseract) ALPR pipeline.
    Yields real-time SSE updates for live dual video canvas previews, vehicle logs, plate crops, and metrics.
    """
    # 1. Reset CSV logging files and clear old crops
    init_log("outputs/logs/detections_easyocr.csv")
    init_log("outputs/logs/detections_pytesseract.csv")
    init_log("outputs/logs/detections.csv")

    easyocr_tracker.flush_all()
    pytesseract_tracker.flush_all()

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        yield f"event: error\ndata: {json.dumps({'error': 'Cannot open video file'})}\n\n"
        return

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    if frames_per_second == 0 or frames_per_second is None:
        frames_per_second = 30.0

    width, height = int(video_capture.get(3)), int(video_capture.get(4))
    if width > 1920:
        scale = 1920 / width
        width, height = 1920, int(height * scale)

    # 2. Configure dedicated OCR Worker thread pools for both engines
    from concurrent.futures import ThreadPoolExecutor
    easyocr_pool = ThreadPoolExecutor(max_workers=2)
    pytesseract_pool = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 2, 6))

    pending_easyocr_futures = []
    pending_pytesseract_futures = []

    # Configure Video Writer output streams
    os.makedirs("outputs/results", exist_ok=True)
    raw_video_easy_path = "outputs/results/raw_output_easyocr.mp4"
    raw_video_tess_path = "outputs/results/raw_output_pytesseract.mp4"

    output_fps = frames_per_second
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer_easy = cv2.VideoWriter(raw_video_easy_path, fourcc, output_fps, (width, height))
    video_writer_tess = cv2.VideoWriter(raw_video_tess_path, fourcc, output_fps, (width, height))

    start_time = time.time()
    frame_idx = 0
    processed_frames_count = 0

    sent_logs = {"EasyOCR": {}, "PyTesseract": {}}
    sent_crops = {"EasyOCR": {}, "PyTesseract": {}}
    sent_texts = {"EasyOCR": {}, "PyTesseract": {}}

    batch_size = 4 if torch.cuda.is_available() else 1
    current_skip = 1

    while True:
        batch_frames = []
        batch_indices = []

        for _ in range(batch_size):
            for _ in range(current_skip - 1):
                video_capture.grab()
                frame_idx += 1

            ret, frame = video_capture.read()
            if not ret:
                break
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            frame_idx += 1

        if not batch_frames:
            break

        batch_start_time = time.time()
        processed_easy_frames, processed_tess_frames = process_batch_dual(
            batch_frames,
            batch_indices,
            easyocr_pool,
            pytesseract_pool,
            pending_easyocr_futures,
            pending_pytesseract_futures,
            fps=frames_per_second
        )

        for p_easy, p_tess in zip(processed_easy_frames, processed_tess_frames):
            for _ in range(current_skip):
                video_writer_easy.write(p_easy)
                video_writer_tess.write(p_tess)
            processed_frames_count += 1

        batch_elapsed = time.time() - batch_start_time
        time_per_frame = batch_elapsed / len(batch_frames)

        native_frame_time = 1.0 / frames_per_second
        speed_based_skip = max(1, int(time_per_frame / native_frame_time))

        max_displacement = max(easyocr_tracker.get_max_displacement(), pytesseract_tracker.get_max_displacement())
        if not easyocr_tracker.tracks and not pytesseract_tracker.tracks:
            velocity_based_skip = 3
        elif max_displacement < 5:
            velocity_based_skip = 8
        elif max_displacement < 15:
            velocity_based_skip = 5
        elif max_displacement < 30:
            velocity_based_skip = 3
        else:
            velocity_based_skip = 1

        if frame_skip == "dynamic":
            current_skip = max(speed_based_skip, velocity_based_skip)
            current_skip = min(current_skip, int(frames_per_second))
        else:
            try:
                current_skip = max(1, int(frame_skip))
            except (ValueError, TypeError):
                current_skip = 1

        if processed_frames_count > 0:
            elapsed_time = time.time() - start_time
            elapsed_str = _format_elapsed(elapsed_time)
            latest_frame_idx = batch_indices[-1]
            progress_percent = int((latest_frame_idx / total_frames) * 100) if total_frames else 0

            eta_str = "--"
            if latest_frame_idx > 0 and total_frames > latest_frame_idx:
                eta_seconds = (elapsed_time / latest_frame_idx) * (total_frames - latest_frame_idx)
                eta_str = _format_elapsed(eta_seconds)

            processing_fps = round(processed_frames_count / elapsed_time, 1) if elapsed_time > 0 else 0

            progress_data = {
                "frame_idx": latest_frame_idx,
                "total_frames": total_frames,
                "percent": progress_percent,
                "elapsed_str": elapsed_str,
                "eta": eta_str,
                "fps": processing_fps
            }
            yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"

            # Stream dual canvas previews (EasyOCR & PyTesseract)
            last_easy = processed_easy_frames[-1]
            last_tess = processed_tess_frames[-1]
            image_h, image_w = last_easy.shape[:2]
            scale = 480.0 / image_w if image_w > 480 else 1.0

            if scale < 1.0:
                last_easy = cv2.resize(last_easy, (int(image_w * scale), int(image_h * scale)))
                last_tess = cv2.resize(last_tess, (int(image_w * scale), int(image_h * scale)))

            _, buffer_easy = cv2.imencode('.jpg', last_easy)
            _, buffer_tess = cv2.imencode('.jpg', last_tess)

            base_easy = base64.b64encode(buffer_easy).decode('utf-8')
            base_tess = base64.b64encode(buffer_tess).decode('utf-8')

            yield f"event: frame\ndata: {json.dumps({'image_easyocr': base_easy, 'image_pytesseract': base_tess})}\n\n"

            # Check and stream CSV log updates & cropped plate images for both models
            gt_plates = load_ground_truth()
            for model_name, csv_path in [("EasyOCR", "outputs/logs/detections_easyocr.csv"), ("PyTesseract", "outputs/logs/detections_pytesseract.csv")]:
                try:
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        for _, row in df.iterrows():
                            plate = str(row['plate_number']) if pd.notna(row['plate_number']) else ""
                            track_id_str = str(row['track_id'])
                            try:
                                track_id = int(row['track_id'])
                            except (ValueError, TypeError):
                                track_id = track_id_str
                            conf = float(row['confidence']) if pd.notna(row['confidence']) else 0.0

                            dedup_key = plate if plate else f"track_{track_id}"
                            prev_log_conf = sent_logs[model_name].get(dedup_key, -1.0)

                            if conf > prev_log_conf:
                                sent_logs[model_name][dedup_key] = conf
                                row_dict = row.to_dict()
                                row_dict["model"] = model_name
                                snapshot_path = row_dict.get("snapshot_path")
                                crop_path = row_dict.get("plate_crop_path")
                                row_dict["snapshot_url"] = "/" + str(snapshot_path).replace("\\", "/") if pd.notna(snapshot_path) and snapshot_path else None
                                row_dict["plate_crop_url"] = "/" + str(crop_path).replace("\\", "/") if pd.notna(crop_path) and crop_path else None

                                if gt_plates and plate:
                                    best_gt, best_cer = find_best_ground_truth_match(plate, gt_plates)
                                    row_dict["cer"] = round(best_cer, 4) if best_cer is not None else None
                                    row_dict["matched_gt"] = best_gt
                                else:
                                    row_dict["cer"] = None
                                    row_dict["matched_gt"] = None

                                yield f"event: log\ndata: {json.dumps(row_dict)}\n\n"

                            prev_crop_conf = sent_crops[model_name].get(track_id, -1.0)
                            prev_text = sent_texts[model_name].get(track_id, "")
                            curr_text = str(row["plate_number"]) if pd.notna(row["plate_number"]) else ""

                            if conf > prev_crop_conf or curr_text != prev_text:
                                sent_crops[model_name][track_id] = conf
                                sent_texts[model_name][track_id] = curr_text

                                crop_path = row.get("plate_crop_path")
                                if pd.notna(crop_path) and crop_path:
                                    crop_url = "/" + str(crop_path).replace("\\", "/")
                                    filename = os.path.basename(str(crop_path))
                                    snapshot_path = row.get("snapshot_path")
                                    snapshot_url = "/" + str(snapshot_path).replace("\\", "/") if pd.notna(snapshot_path) and snapshot_path else None

                                    crop_payload = {
                                        'model': model_name,
                                        'filename': filename,
                                        'url': crop_url,
                                        'text': curr_text,
                                        'track_id': track_id,
                                        'snapshot_url': snapshot_url,
                                        'confidence': conf,
                                        'vehicle_type': str(row["vehicle_type"]) if pd.notna(row["vehicle_type"]) else None,
                                        'color': str(row["color"]) if pd.notna(row["color"]) else None,
                                        'timestamp': str(row["timestamp"]) if pd.notna(row["timestamp"]) else None
                                    }
                                    yield f"event: crop\ndata: {json.dumps(crop_payload)}\n\n"
                except Exception:
                    pass

        await asyncio.sleep(0.001)

    easyocr_tracker.purge_old(frame_idx=frame_idx, frames_per_second=frames_per_second, force_flush=True)
    pytesseract_tracker.purge_old(frame_idx=frame_idx, frames_per_second=frames_per_second, force_flush=True)

    drain_pending_ocr(pending_easyocr_futures, target_tracker=easyocr_tracker)
    drain_pending_ocr(pending_pytesseract_futures, target_tracker=pytesseract_tracker)

    easyocr_pool.shutdown(wait=True)
    pytesseract_pool.shutdown(wait=True)

    video_capture.release()
    video_writer_easy.release()
    video_writer_tess.release()

    # Transcode both videos to H.264 mp4
    final_easy_path = "outputs/results/final_output_easyocr.mp4"
    final_tess_path = "outputs/results/final_output_pytesseract.mp4"

    for raw_p, final_p in [(raw_video_easy_path, final_easy_path), (raw_video_tess_path, final_tess_path)]:
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", raw_p,
                    "-vcodec", "libx264",
                    "-preset", "veryfast",
                    "-crf", "28",
                    "-movflags", "+faststart",
                    final_p
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    final_elapsed = time.time() - start_time
    final_fps = round(processed_frames_count / final_elapsed, 1) if final_elapsed > 0 else 0

    # Load detections and compute comparative Ground Truth metrics
    gt_plates = load_ground_truth()
    easy_dets = pd.read_csv("outputs/logs/detections_easyocr.csv").to_dict(orient="records") if os.path.exists("outputs/logs/detections_easyocr.csv") else []
    tess_dets = pd.read_csv("outputs/logs/detections_pytesseract.csv").to_dict(orient="records") if os.path.exists("outputs/logs/detections_pytesseract.csv") else []

    comparison_metrics = compute_dual_model_comparison(easy_dets, tess_dets, gt_plates, easyocr_time=final_elapsed, pytesseract_time=final_elapsed)

    yield f"event: progress\ndata: {json.dumps({'frame_idx': total_frames, 'total_frames': total_frames, 'percent': 100, 'elapsed_str': _format_elapsed(final_elapsed), 'eta': 'Done', 'fps': final_fps})}\n\n"

    complete_payload = {
        'video_easyocr_url': '/outputs/results/final_output_easyocr.mp4' if os.path.exists(final_easy_path) else '/outputs/results/raw_output_easyocr.mp4',
        'video_pytesseract_url': '/outputs/results/final_output_pytesseract.mp4' if os.path.exists(final_tess_path) else '/outputs/results/raw_output_pytesseract.mp4',
        'metrics': comparison_metrics
    }
    yield f"event: complete\ndata: {json.dumps(complete_payload)}\n\n"


# ─── FastAPI App & Custom Dashboard Serving ───────────────────────

app = FastAPI(title="ALPR & Vehicle Classification - Dual Model Comparison")

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("src/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/upload")
async def upload_video_api(file: UploadFile = File(...)):
    os.makedirs("outputs/uploads", exist_ok=True)
    filepath = f"outputs/uploads/{file.filename}"
    with open(filepath, "wb") as f:
        f.write(await file.read())
    return {"filepath": filepath}

@app.get("/api/stream-process")
async def stream_process_api(video_path: str, ocr_engine: str = "dual", frame_skip: str = "dynamic"):
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        process_video_sse(video_path, ocr_engine, frame_skip),
        media_type="text/event-stream",
        headers=headers
    )


# ─── Ground Truth & Comparison API ───────────────────────────────

@app.get("/api/ground-truth")
async def get_ground_truth():
    plates = load_ground_truth()
    return JSONResponse({"plates": plates})

@app.post("/api/ground-truth")
async def set_ground_truth(request: Request):
    body = await request.json()
    plates = body.get("plates", [])
    cleaned = list(dict.fromkeys(p.strip() for p in plates if p.strip()))
    save_ground_truth(cleaned)
    return JSONResponse({"status": "ok", "count": len(cleaned), "plates": cleaned})

@app.post("/api/ground-truth/upload")
async def upload_ground_truth(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8", errors="ignore")
    plates = []
    
    if file.filename and file.filename.lower().endswith(".csv"):
        import csv
        import io
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            plate = row.get("plate_number") or row.get("plate") or row.get("Plate Number") or row.get("Plate") or ""
            if plate.strip():
                plates.append(plate.strip())
    else:
        for line in content.splitlines():
            cleaned = line.strip()
            if cleaned:
                plates.append(cleaned)
    
    cleaned = list(dict.fromkeys(plates))
    save_ground_truth(cleaned)
    return JSONResponse({"status": "ok", "count": len(cleaned), "plates": cleaned})

@app.get("/api/cer-summary")
async def cer_summary_api():
    """
    Returns full side-by-side comparative Ground Truth statistics for EasyOCR vs PyTesseract.
    """
    gt_plates = load_ground_truth()
    easy_csv = "outputs/logs/detections_easyocr.csv"
    tess_csv = "outputs/logs/detections_pytesseract.csv"

    easy_dets = pd.read_csv(easy_csv).to_dict(orient="records") if os.path.exists(easy_csv) else []
    tess_dets = pd.read_csv(tess_csv).to_dict(orient="records") if os.path.exists(tess_csv) else []

    comparison = compute_dual_model_comparison(easy_dets, tess_dets, gt_plates)
    return JSONResponse(comparison)

@app.get("/api/export")
async def export_api():
    """Generates and downloads the Excel multi-tab comparison report."""
    easy_csv = "outputs/logs/detections_easyocr.csv"
    tess_csv = "outputs/logs/detections_pytesseract.csv"

    if not os.path.exists(easy_csv) and not os.path.exists(tess_csv):
        return HTMLResponse(content="<h3>No records available to export yet.</h3>", status_code=400)
    
    try:
        from src.export import build_xlsx_report
    except ImportError:
        return HTMLResponse(content="<h3>openpyxl is not installed. Please add it to requirements.txt and install it.</h3>", status_code=500)
    
    gt_plates = load_ground_truth()
    
    try:
        easy_df = pd.read_csv(easy_csv) if os.path.exists(easy_csv) else pd.DataFrame()
        tess_df = pd.read_csv(tess_csv) if os.path.exists(tess_csv) else pd.DataFrame()
        out_buf = build_xlsx_report(easy_df, tess_df, gt_plates)
    except Exception as e:
        return HTMLResponse(content=f"<h3>Failed to build export: {str(e)}</h3>", status_code=500)
        
    return StreamingResponse(
        out_buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=alpr_dual_model_comparison.xlsx"}
    )


# ─── Server Entry Point ──────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

