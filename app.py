import os
import time
import asyncio
import base64
import json

import cv2
import torch
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from src.pipeline import vehicle_tracker, process_batch_dual, drain_pending_ocr
from src.logging.logger import init_log
from src.metrics.cer import (
    find_best_ground_truth_match, save_ground_truth, load_ground_truth,
    compute_dual_model_comparison
)

def _format_elapsed(seconds):
    """
    Formats raw execution time in seconds into a friendly human-readable 'Mm Ss' string.
    """
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ─── Server-Sent Events (SSE) Real-Time Generator ────────────────

def _compute_frame_skip(batch_elapsed, batch_size, fps, frame_skip_mode):
    """
    Calculates optimal frame skip based on processing speed and vehicle velocity.
    Returns the number of frames to skip between processed frames.
    """
    time_per_frame = batch_elapsed / batch_size
    native_frame_time = 1.0 / fps
    speed_based_skip = max(1, int(time_per_frame / native_frame_time))

    max_displacement = vehicle_tracker.get_max_displacement()
    if not vehicle_tracker.tracks:
        velocity_based_skip = 3
    elif max_displacement < 5:
        velocity_based_skip = 8
    elif max_displacement < 15:
        velocity_based_skip = 5
    elif max_displacement < 30:
        velocity_based_skip = 3
    else:
        velocity_based_skip = 1

    if frame_skip_mode == "dynamic":
        current_skip = max(speed_based_skip, velocity_based_skip)
        return min(current_skip, int(fps))
    try:
        return max(1, int(frame_skip_mode))
    except (ValueError, TypeError):
        return 1


def _build_detection_events(sent_logs, sent_crops, sent_texts, gt_plates):
    """
    Reads detection CSVs and returns SSE event strings for new/updated detections.
    Handles both log events (detection telemetry) and crop events (plate crop images).
    """
    events = []
    for model_name, csv_path in [("EasyOCR", "outputs/logs/detections_easyocr.csv"), ("PyTesseract", "outputs/logs/detections_pytesseract.csv")]:
        try:
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                plate = str(row['plate_number']) if pd.notna(row['plate_number']) else ""
                track_id_str = str(row['track_id'])
                try:
                    track_id = int(row['track_id'])
                except (ValueError, TypeError):
                    track_id = track_id_str
                conf = float(row['confidence']) if pd.notna(row['confidence']) else 0.0

                if gt_plates and plate:
                    best_gt, best_cer = find_best_ground_truth_match(plate, gt_plates)
                else:
                    best_gt, best_cer = None, None

                norm_gt = best_gt.replace(" ", "").upper() if best_gt else ""
                norm_plate = plate.replace(" ", "").upper() if plate else ""
                dedup_key = f"gt_{norm_gt}" if norm_gt else (norm_plate if norm_plate else f"track_{track_id}")

                prev_log_conf = sent_logs[model_name].get(dedup_key, -1.0)

                if conf > prev_log_conf:
                    sent_logs[model_name][dedup_key] = conf
                    row_dict = row.to_dict()
                    row_dict["model"] = model_name
                    snapshot_path = row_dict.get("snapshot_path")
                    crop_path = row_dict.get("plate_crop_path")
                    row_dict["snapshot_url"] = "/" + str(snapshot_path).replace("\\", "/") if pd.notna(snapshot_path) and snapshot_path else None
                    row_dict["plate_crop_url"] = "/" + str(crop_path).replace("\\", "/") if pd.notna(crop_path) and crop_path else None
                    row_dict["cer"] = round(best_cer, 4) if best_cer is not None else None
                    row_dict["matched_gt"] = best_gt if best_gt else "--"
                    events.append(f"event: log\ndata: {json.dumps(row_dict)}\n\n")

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
                        events.append(f"event: crop\ndata: {json.dumps(crop_payload)}\n\n")
        except Exception:
            pass
    return events

async def process_video_sse(video_path, ocr_engine="dual", frame_skip="dynamic"):
    """
    Asynchronous generator streaming real-time Server-Sent Events (SSE) to the browser.
    
    Processes the uploaded video using dual models (EasyOCR + PyTesseract) simultaneously.
    Yields live frame previews, vehicle detection telemetry logs, plate crop images, and accuracy metrics.
    """
    try:
        # Step 1: Initialize detection CSV log files and reset track memory states
        init_log("outputs/logs/detections_easyocr.csv")
        init_log("outputs/logs/detections_pytesseract.csv")

        vehicle_tracker.flush_all()

        # Step 2: Open input video stream using OpenCV VideoCapture
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            yield f"event: error\ndata: {json.dumps({'error': 'Cannot open video file'})}\n\n"
            return

        # Extract video metadata (total frame count and FPS frame rate)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
        if frames_per_second == 0 or frames_per_second is None:
            frames_per_second = 30.0

        width, height = int(video_capture.get(3)), int(video_capture.get(4))
        if width > 1920:
            scale = 1920 / width
            width, height = 1920, int(height * scale)

        os.makedirs("outputs", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_easy = cv2.VideoWriter("outputs/annotated_easyocr.mp4", fourcc, frames_per_second, (width, height))
        out_tess = cv2.VideoWriter("outputs/annotated_pytesseract.mp4", fourcc, frames_per_second, (width, height))

        from concurrent.futures import ThreadPoolExecutor
        easyocr_pool = ThreadPoolExecutor(max_workers=2)
        pytesseract_pool = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 2, 6))

        pending_easyocr_futures = []
        pending_pytesseract_futures = []

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
            batch_intermediate_frames = []

            for _ in range(batch_size):
                skipped_frames = []
                for _ in range(current_skip - 1):
                    ret_s, f_s = video_capture.read()
                    if ret_s:
                        f_s = cv2.resize(f_s, (width, height))
                        skipped_frames.append(f_s)
                        frame_idx += 1

                ret, frame = video_capture.read()
                if not ret:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                batch_intermediate_frames.append(skipped_frames)
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

            for fe, ft in zip(processed_easy_frames, processed_tess_frames):
                out_easy.write(fe)
                out_tess.write(ft)

            processed_frames_count += len(processed_easy_frames)

            batch_elapsed = time.time() - batch_start_time
            current_skip = _compute_frame_skip(batch_elapsed, len(batch_frames), frames_per_second, frame_skip)

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

                gt_plates = load_ground_truth()
                for event_str in _build_detection_events(sent_logs, sent_crops, sent_texts, gt_plates):
                    yield event_str

            await asyncio.sleep(0.001)

        vehicle_tracker.purge_old(frame_idx=frame_idx, frames_per_second=frames_per_second, force_flush=True)

        drain_pending_ocr(pending_easyocr_futures)
        drain_pending_ocr(pending_pytesseract_futures)

        easyocr_pool.shutdown(wait=True)
        pytesseract_pool.shutdown(wait=True)

        out_easy.release()
        out_tess.release()
        video_capture.release()

        final_elapsed = time.time() - start_time
        final_fps = round(processed_frames_count / final_elapsed, 1) if final_elapsed > 0 else 0

        gt_plates = load_ground_truth()
        easy_dets = pd.read_csv("outputs/logs/detections_easyocr.csv").to_dict(orient="records") if os.path.exists("outputs/logs/detections_easyocr.csv") else []
        tess_dets = pd.read_csv("outputs/logs/detections_pytesseract.csv").to_dict(orient="records") if os.path.exists("outputs/logs/detections_pytesseract.csv") else []

        comparison_metrics = compute_dual_model_comparison(easy_dets, tess_dets, gt_plates, easyocr_time=final_elapsed, pytesseract_time=final_elapsed)

        yield f"event: progress\ndata: {json.dumps({'frame_idx': total_frames, 'total_frames': total_frames, 'percent': 100, 'elapsed_str': _format_elapsed(final_elapsed), 'eta': 'Done', 'fps': final_fps})}\n\n"

        complete_payload = {
            'metrics': comparison_metrics,
            'video_easyocr_url': '/outputs/annotated_easyocr.mp4',
            'video_pytesseract_url': '/outputs/annotated_pytesseract.mp4'
        }
        yield f"event: complete\ndata: {json.dumps(complete_payload)}\n\n"

    except Exception as e:
        import traceback
        print("\n" + "="*50)
        print("[ALPR PIPELINE ERROR DETECTED]")
        traceback.print_exc()
        print("="*50 + "\n")
        yield f"event: error_msg\ndata: {json.dumps({'error': str(e)})}\n\n"



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
    try:
        os.makedirs("outputs/uploads", exist_ok=True)
        filename = file.filename if file.filename else "uploaded_video.mp4"
        filepath = f"outputs/uploads/{filename}"
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        return {"filepath": filepath}
    except Exception as e:
        import traceback
        print(f"[UPLOAD ERROR] {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

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


# ─── Server Entry Point ──────────────────────────────────────────

def find_available_port(start_port=7860, max_attempts=20):
    import socket
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return start_port


if __name__ == "__main__":
    port = find_available_port(7860)
    print(f"🚀 Starting ALPR Dual-Model Dashboard on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


