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
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sys

class LogMirror:
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, data):
        self.original_stream.write(data)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(data)
        except Exception:
            pass

    def flush(self):
        self.original_stream.flush()

os.makedirs("outputs/logs", exist_ok=True)
server_log_path = "outputs/logs/server.log"
try:
    with open(server_log_path, "w", encoding="utf-8") as f:
        f.write("--- Server Log Started ---\n")
except Exception:
    pass

sys.stdout = LogMirror(sys.stdout, server_log_path)
sys.stderr = LogMirror(sys.stderr, server_log_path)

import torch
from src.pipeline import FRAME_SKIP, detection_tracker, process_frame, process_batch, drain_pending_ocr
from src.logging.logger import init_log

def _format_elapsed(seconds):
    """Format elapsed seconds into a readable string."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ─── Server-Sent Events (SSE) Real-Time Generator ────────────────

async def process_video_sse(video_path, ocr_engine):
    # Reset CSV logs and clear crops/snapshots
    init_log()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield f"event: error\ndata: {json.dumps({'error': 'Cannot open video file'})}\n\n"
        return

    # Extract video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30

    width, height = int(cap.get(3)), int(cap.get(4))
    # Cap processing and output size at 1080p
    if width > 1920:
        scale = 1920 / width
        width, height = 1920, int(height * scale)

    # Initialize session-dedicated ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    if ocr_engine == "PyTesseract":
        workers = min(os.cpu_count() or 2, 6)
    else:
        workers = 2
    ocr_pool = ThreadPoolExecutor(max_workers=workers)
    pending_futures = []

    os.makedirs("outputs/results", exist_ok=True)
    out_path = "outputs/results/processed_output.mp4"
    output_fps = fps / 3.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, output_fps, (width, height))

    start_time = time.time()
    frame_idx = 0
    processed_count = 0

    sent_logs = set()
    sent_crops = {}  # track_id -> best confidence sent so far

    # Set micro-batching parameters
    batch_size = 4 if torch.cuda.is_available() else 1
    current_skip = 1  # Spacing skip between processed frames (starts at 1)

    while True:
        batch_frames = []
        batch_indices = []

        # Read frames with current spacing skip in micro-batch
        for _ in range(batch_size):
            # Skip current_skip - 1 frames using grab()
            for _ in range(current_skip - 1):
                cap.grab()
                frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            frame_idx += 1

        if not batch_frames:
            break

        # Process the micro-batch and measure elapsed time
        batch_start_time = time.time()
        processed_frames = process_batch(
            batch_frames,
            batch_indices,
            ocr_engine,
            ocr_pool,
            pending_futures,
            fps=fps
        )

        for p_frame in processed_frames:
            out.write(p_frame)
            processed_count += 1

        # Calculate time taken per frame in this batch
        batch_elapsed = time.time() - batch_start_time
        t_frame = batch_elapsed / len(batch_frames)

        # Dynamic Skip Factor 1: speed-based target close to real-time
        t_native = 1.0 / fps
        speed_skip = max(1, int(t_frame / t_native))

        # Dynamic Skip Factor 2: vehicle velocity based
        max_d = detection_tracker.get_max_displacement()
        if not detection_tracker.tracks:
            velocity_skip = 3
        elif max_d < 5:
            velocity_skip = 8  # Static or very slow: skip more frames
        elif max_d < 15:
            velocity_skip = 5
        elif max_d < 30:
            velocity_skip = 3
        else:
            velocity_skip = 1  # Fast moving: reduce skip to maintain tracking

        # Set dynamic skip spacing for next batch
        current_skip = max(speed_skip, velocity_skip)
        current_skip = min(current_skip, int(fps))  # Cap skip to 1 second maximum

        # Send state updates periodically
        if processed_count > 0:
            elapsed = time.time() - start_time
            elapsed_str = _format_elapsed(elapsed)
            latest_frame_idx = batch_indices[-1]
            progress_val = int((latest_frame_idx / total_frames) * 100) if total_frames else 0

            eta_str = "--"
            if progress_val > 0:
                eta_seconds = (elapsed / progress_val) * (100 - progress_val)
                eta_str = _format_elapsed(eta_seconds)

            fps_val = round(processed_count / elapsed, 1) if elapsed > 0 else 0

            # Yield progress stats
            progress_data = {
                "frame_idx": latest_frame_idx,
                "total_frames": total_frames,
                "percent": progress_val,
                "elapsed_str": elapsed_str,
                "eta": eta_str,
                "fps": fps_val
            }
            yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"

            # Stream canvas preview frame
            last_processed = processed_frames[-1]
            h_img, w_img = last_processed.shape[:2]
            scale_img = 480.0 / w_img if w_img > 480 else 1.0
            preview_frame = last_processed
            if scale_img < 1.0:
                preview_frame = cv2.resize(last_processed, (int(w_img * scale_img), int(h_img * scale_img)))

            _, buffer = cv2.imencode('.jpg', preview_frame)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            yield f"event: frame\ndata: {json.dumps({'image': img_b64})}\n\n"

            # Check and stream CSV log changes
            try:
                df = pd.read_csv("outputs/logs/detections.csv")
                for _, row in df.iterrows():
                    track_key = f"{row['track_id']}_{row['confidence']}"
                    if track_key not in sent_logs:
                        sent_logs.add(track_key)
                        row_dict = row.to_dict()
                        snap_path = row_dict.get("snapshot_path")
                        crop_path = row_dict.get("plate_crop_path")
                        row_dict["snapshot_url"] = "/" + str(snap_path) if pd.notna(snap_path) and snap_path else None
                        row_dict["plate_crop_url"] = "/" + str(crop_path) if pd.notna(crop_path) and crop_path else None
                        yield f"event: log\ndata: {json.dumps(row_dict)}\n\n"
            except Exception:
                pass

            # Check and stream newly cropped plate images
            try:
                crop_files = glob.glob("outputs/plate_crops/Processed/*.jpg")
                for filepath in crop_files:
                    filename = os.path.basename(filepath)
                    try:
                        track_id = int(filename.split("track")[1].split("_")[0])
                    except (IndexError, ValueError):
                        continue

                    text_read = ""
                    current_conf = 0.0
                    snapshot_url = None
                    vehicle_type = None
                    color = None
                    timestamp = None
                    try:
                        match = df[df["track_id"] == track_id]
                        if not match.empty:
                            row = match.iloc[0]
                            text_read = str(row["plate_number"]) if pd.notna(row["plate_number"]) else ""
                            current_conf = float(row["confidence"])
                            snap_val = row.get("snapshot_path")
                            if pd.notna(snap_val) and snap_val:
                                snapshot_url = "/" + str(snap_val)
                            if pd.notna(row.get("vehicle_type")):
                                vehicle_type = str(row["vehicle_type"])
                            if pd.notna(row.get("color")):
                                color = str(row["color"])
                            if pd.notna(row.get("timestamp")):
                                timestamp = str(row["timestamp"])
                    except Exception:
                        pass

                    prev_conf = sent_crops.get(track_id, -1)
                    if current_conf > prev_conf:
                        sent_crops[track_id] = current_conf
                        crop_payload = {
                            'filename': filename,
                            'url': f'/outputs/plate_crops/Processed/{filename}',
                            'text': text_read,
                            'track_id': track_id,
                            'snapshot_url': snapshot_url,
                            'confidence': current_conf,
                            'vehicle_type': vehicle_type,
                            'color': color,
                            'timestamp': timestamp
                        }
                        yield f"event: crop\ndata: {json.dumps(crop_payload)}\n\n"
            except Exception:
                pass

        # Relinquish CPU execution control back to asyncio event loop
        await asyncio.sleep(0.001)

    # Force sweep remaining active tracks and write their final data to the logs
    detection_tracker.purge_old(frame_idx=frame_idx, fps=fps, force_flush=True)

    # Drain OCR threads and flush logs
    drain_pending_ocr(pending_futures)
    ocr_pool.shutdown(wait=True)
    detection_tracker.flush_all()
    cap.release()
    out.release()

    # Transcode final output to H.264 browser format
    final_out_path = "outputs/results/final_output.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_path, "-vcodec", "libx264", "-preset", "ultrafast", final_out_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        final_out_path = out_path

    # Send final progress event so the frames counter matches exactly
    final_elapsed = time.time() - start_time
    final_fps = round(processed_count / final_elapsed, 1) if final_elapsed > 0 else 0
    yield f"event: progress\ndata: {json.dumps({'frame_idx': total_frames, 'total_frames': total_frames, 'percent': 100, 'elapsed_str': _format_elapsed(final_elapsed), 'eta': 'Done', 'fps': final_fps})}\n\n"

    yield f"event: complete\ndata: {json.dumps({'video_url': '/outputs/results/final_output.mp4'})}\n\n"





# ─── FastAPI App & Custom Dashboard Serving ───────────────────────

# Create the primary FastAPI application (we own the server)
app = FastAPI(title="ALPR & Vehicle Classification")

# Mount output storage locally
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Serve custom dashboard at root "/"
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("src/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Register video file upload handler API
@app.post("/api/upload")
async def upload_video_api(file: UploadFile = File(...)):
    os.makedirs("outputs/uploads", exist_ok=True)
    filepath = f"outputs/uploads/{file.filename}"
    with open(filepath, "wb") as f:
        f.write(await file.read())
    return {"filepath": filepath}

# Register Server-Sent Events (SSE) stream processing endpoint
@app.get("/api/stream-process")
async def stream_process_api(video_path: str, ocr_engine: str):
    return StreamingResponse(
        process_video_sse(video_path, ocr_engine),
        media_type="text/event-stream"
    )

# Register Logs API endpoint
@app.get("/api/logs")
async def get_logs_api():
    server_log_path = "outputs/logs/server.log"
    # Fallback to Colab log if it exists and server.log is empty
    if not os.path.exists(server_log_path):
        if os.path.exists("uvicorn.log"):
            server_log_path = "uvicorn.log"
        else:
            return {"logs": "No logs available."}
    try:
        with open(server_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return {"logs": "".join(lines[-200:])}
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}"}

# Register Export API endpoint (CSV or XLSX with embedded images)
@app.get("/api/export")
async def export_api(format: str = "csv"):
    csv_path = "outputs/logs/detections.csv"
    if not os.path.exists(csv_path):
        return HTMLResponse(content="<h3>No records available to export yet.</h3>", status_code=400)
    
    if format == "csv":
        try:
            from io import BytesIO
            df = pd.read_csv(csv_path)
            # Ensure columns exist, if not create empty ones
            for col in ["track_id", "timestamp", "vehicle_type", "color", "plate_number", "confidence", "snapshot_path", "plate_crop_path"]:
                if col not in df.columns:
                    df[col] = ""
            
            # Map paths to relative web URLs
            def to_url(val):
                if pd.isna(val) or not val:
                    return ""
                val_str = str(val).strip()
                if val_str.startswith("outputs/"):
                    return "/" + val_str
                return val_str

            df["snapshot_url"] = df["snapshot_path"].apply(to_url)
            df["plate_crop_url"] = df["plate_crop_path"].apply(to_url)
            
            # Select and rename columns
            export_df = df[[
                "track_id", "timestamp", "vehicle_type", "color", 
                "plate_number", "confidence", "snapshot_url", "plate_crop_url"
            ]].copy()
            
            export_df.columns = [
                "ID", "Timestamp", "Type", "Color", 
                "Plate Number", "Confidence", "Vehicle Image", "Corresponding Plate Image"
            ]
            
            # Format confidence as percentage string or float
            export_df["Confidence"] = export_df["Confidence"].apply(lambda x: f"{float(x)*100:.0f}%" if pd.notna(x) and x != "" else "")
            
            # Capitalize vehicle type and color
            export_df["Type"] = export_df["Type"].apply(lambda x: str(x).capitalize() if pd.notna(x) else "")
            export_df["Color"] = export_df["Color"].apply(lambda x: str(x).capitalize() if pd.notna(x) else "")
            
            # Write to buffer
            out_buf = BytesIO()
            export_df.to_csv(out_buf, index=False)
            out_buf.seek(0)
            
            return StreamingResponse(
                out_buf,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=alpr_detections.csv"}
            )
        except Exception as e:
            return HTMLResponse(content=f"<h3>Failed to build CSV export: {str(e)}</h3>", status_code=500)
        
    elif format == "xlsx":
        try:
            from src.export import build_xlsx_report
        except ImportError:
            return HTMLResponse(content="<h3>openpyxl is not installed. Please add it to requirements.txt and install it.</h3>", status_code=500)
        
        try:
            df = pd.read_csv(csv_path)
            out_buf = build_xlsx_report(df)
        except Exception as e:
            return HTMLResponse(content=f"<h3>Failed to build export: {str(e)}</h3>", status_code=500)
            
        return StreamingResponse(
            out_buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=alpr_detections.xlsx"}
        )
        
    return HTMLResponse(content="<h3>Invalid export format. Only 'csv' and 'xlsx' are supported.</h3>", status_code=400)


# ─── Server Entry Point ──────────────────────────────────────────

if __name__ == "__main__":
    # Default: run locally on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
