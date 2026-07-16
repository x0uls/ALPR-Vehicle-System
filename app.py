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
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import sys

class LogMirror:
    """
    Redirects stdout/stderr streams to a file on disk while mirroring them back to the terminal.
    
    This ensures we write background execution history to a server.log file that can be displayed
    on the web dashboard front-end without losing the console output.
    """
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

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

# Redirect program console printouts to server.log for dashboard log panels
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
from src.pipeline import detection_tracker, process_batch, drain_pending_ocr
from src.logging.logger import init_log

def _format_elapsed(seconds):
    """
    Formats elapsed raw seconds into a friendly human-readable time string.
    
    For example: 95 seconds becomes '1m 35s'.
    """
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ─── Server-Sent Events (SSE) Real-Time Generator ────────────────

async def process_video_sse(video_path, ocr_engine):
    """
    Asynchronous generator that runs the video detection pipeline and yields real-time updates.
    
    Uses SSE format ('event: ...\ndata: ...\n\n') to stream live dashboard stats, log events,
    and base64-encoded frame images back to the user's web browser page.
    """
    # 1. Reset CSV logging files and remove old snapshot images
    init_log()

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        yield f"event: error\ndata: {json.dumps({'error': 'Cannot open video file'})}\n\n"
        return

    # Extract video properties
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    if frames_per_second == 0 or frames_per_second is None:
        frames_per_second = 30  # Default fallback standard framerate if metadata query fails

    width, height = int(video_capture.get(3)), int(video_capture.get(4))
    # Downscale resolution to 1080p if video width exceeds 1920px.
    # Prevents huge images from thrasher memory or slowing down YOLO processing.
    if width > 1920:
        scale = 1920 / width
        width, height = 1920, int(height * scale)

    # 2. Configure dedicated OCR Worker thread pool size
    from concurrent.futures import ThreadPoolExecutor
    if ocr_engine == "PyTesseract":
        # PyTesseract spawns a CLI sub-process command, which is heavy on CPU.
        # Limit worker thread count to avoid high CPU core context-switching overhead.
        worker_threads = min(os.cpu_count() or 2, 6)
    else:
        # EasyOCR runs on GPU/Cuda tensors. EasyOCR handles batches well; 2 threads are plenty
        # to feed the queue without overloading memory buffers.
        worker_threads = 2
    ocr_worker_pool = ThreadPoolExecutor(max_workers=worker_threads)
    pending_ocr_futures = []

    # Configure Video Writer output streams
    os.makedirs("outputs/results", exist_ok=True)
    output_video_path = "outputs/results/processed_output.mp4"
    output_fps = frames_per_second
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))

    start_time = time.time()
    frame_idx = 0
    processed_frames_count = 0

    # Dictionaries to track what data has already been sent to the dashboard to avoid redundant SSE events
    sent_logs = {}    # deduplication key -> best confidence sent so far
    sent_crops = {}   # track_id -> best confidence crop sent so far
    sent_texts = {}   # track_id -> best text reading sent so far

    # 3. Configure micro-batching sizes.
    # If a CUDA GPU is available, process frames in batches of 4 to exploit parallel GPU calculations.
    # On CPU, process 1 frame at a time to prevent high RAM paging/swapping.
    batch_size = 4 if torch.cuda.is_available() else 1
    current_skip = 1  # Spacing skip between processed frames (starts at 1)

    while True:
        batch_frames = []
        batch_indices = []

        # Read frames with current spacing skip in micro-batch
        for _ in range(batch_size):
            # Skip current_skip - 1 frames using grab()
            # grab() is much faster than read() because it discards frames without decoding their visual data
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

        # Process the micro-batch and measure elapsed time
        batch_start_time = time.time()
        processed_frames = process_batch(
            batch_frames,
            batch_indices,
            ocr_engine,
            ocr_worker_pool,
            pending_ocr_futures,
            fps=frames_per_second
        )

        for p_frame in processed_frames:
            # Repeat the annotated frame current_skip times to preserve the output video duration
            for _ in range(current_skip):
                video_writer.write(p_frame)
            processed_frames_count += 1

        # Calculate time taken per frame in this batch
        batch_elapsed = time.time() - batch_start_time
        time_per_frame = batch_elapsed / len(batch_frames)

        # 4. Dynamic Skip Factor 1: speed-based target close to real-time.
        # If processing is slow, skip frames to keep up with the video framerate.
        native_frame_time = 1.0 / frames_per_second
        speed_based_skip = max(1, int(time_per_frame / native_frame_time))

        # Dynamic Skip Factor 2: vehicle velocity based
        max_displacement = detection_tracker.get_max_displacement()
        if not detection_tracker.tracks:
            # No active tracks: skip aggressively (up to 1s of video) to process empty segments faster
            velocity_based_skip = min(15, int(frames_per_second))
        else:
            # Active tracks: keep skip small to maintain continuous tracking overlap
            if max_displacement > 20:
                velocity_based_skip = 1  # Fast moving: process every frame to avoid losing track
            elif max_displacement > 10:
                velocity_based_skip = 2
            else:
                velocity_based_skip = 3  # Slow/static: process every 3rd frame

        # Set dynamic skip spacing for the next batch
        current_skip = max(speed_based_skip, velocity_based_skip)
        current_skip = min(current_skip, int(frames_per_second))  # Cap skip to 1 second maximum

        # Send state updates periodically
        if processed_frames_count > 0:
            elapsed_time = time.time() - start_time
            elapsed_str = _format_elapsed(elapsed_time)
            latest_frame_idx = batch_indices[-1]
            progress_percent = int((latest_frame_idx / total_frames) * 100) if total_frames else 0

            # Calculate ETA (Estimated Time of Arrival)
            eta_str = "--"
            if progress_percent > 0:
                eta_seconds = (elapsed_time / progress_percent) * (100 - progress_percent)
                eta_str = _format_elapsed(eta_seconds)

            processing_fps = round(processed_frames_count / elapsed_time, 1) if elapsed_time > 0 else 0

            # Yield progress stats back to dashboard
            progress_data = {
                "frame_idx": latest_frame_idx,
                "total_frames": total_frames,
                "percent": progress_percent,
                "elapsed_str": elapsed_str,
                "eta": eta_str,
                "fps": processing_fps
            }
            yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"

            # Stream canvas preview frame (downscaled to width 480 to save bandwidth)
            last_processed_frame = processed_frames[-1]
            image_height, image_width = last_processed_frame.shape[:2]
            image_scale = 480.0 / image_width if image_width > 480 else 1.0
            preview_frame = last_processed_frame
            if image_scale < 1.0:
                preview_frame = cv2.resize(last_processed_frame, (int(image_width * image_scale), int(image_height * image_scale)))

            # Encode image to JPEG, convert to base64 string, and yield event
            _, jpeg_buffer = cv2.imencode('.jpg', preview_frame)
            image_base64 = base64.b64encode(jpeg_buffer).decode('utf-8')
            yield f"event: frame\ndata: {json.dumps({'image': image_base64})}\n\n"

            # Check and stream CSV log changes & newly cropped plate images
            try:
                if os.path.exists("outputs/logs/detections.csv"):
                    detections_dataframe = pd.read_csv("outputs/logs/detections.csv")
                    for _, row in detections_dataframe.iterrows():
                        track_id_string = str(row['track_id'])
                        canonical_id = str(row['canonical_id']) if ('canonical_id' in row and pd.notna(row['canonical_id'])) else track_id_string
                        conf = float(row['confidence']) if pd.notna(row['confidence']) else 0.0
                        
                        # 1. Yield event if confidence improved for this canonical track
                        deduplication_key = f"canonical_{canonical_id}"
                        previous_log_confidence = sent_logs.get(deduplication_key, -1.0)
                        if conf > previous_log_confidence:
                            sent_logs[deduplication_key] = conf
                            row_dict = row.to_dict()
                            snapshot_path = row_dict.get("snapshot_path")
                            crop_path = row_dict.get("plate_crop_path")
                            row_dict["snapshot_url"] = "/" + str(snapshot_path) if pd.notna(snapshot_path) and snapshot_path else None
                            row_dict["plate_crop_url"] = "/" + str(crop_path) if pd.notna(crop_path) and crop_path else None
                            yield f"event: log\ndata: {json.dumps(row_dict)}\n\n"
                        
                        # 2. Yield crop update if confidence improved or plate text changed
                        previous_crop_confidence = sent_crops.get(canonical_id, -1.0)
                        previous_text = sent_texts.get(canonical_id, "")
                        current_text = str(row["plate_number"]) if pd.notna(row["plate_number"]) else ""
                        
                        if conf > previous_crop_confidence or current_text != previous_text:
                            sent_crops[canonical_id] = conf
                            sent_texts[canonical_id] = current_text
                            
                            crop_path = row.get("plate_crop_path")
                            if pd.notna(crop_path) and crop_path:
                                crop_url = "/" + str(crop_path).replace("\\", "/")
                                filename = os.path.basename(str(crop_path))
                                
                                snapshot_path = row.get("snapshot_path")
                                snapshot_url = "/" + str(snapshot_path).replace("\\", "/") if pd.notna(snapshot_path) and snapshot_path else None
                                
                                try:
                                    track_id = int(row["track_id"])
                                except (ValueError, TypeError):
                                    track_id = track_id_string
                                
                                crop_payload = {
                                    'filename': filename,
                                    'url': crop_url,
                                    'text': current_text,
                                    'track_id': track_id,
                                    'canonical_id': canonical_id,
                                    'snapshot_url': snapshot_url,
                                    'confidence': conf,
                                    'vehicle_type': str(row["vehicle_type"]) if pd.notna(row["vehicle_type"]) else None,
                                    'color': str(row["color"]) if pd.notna(row["color"]) else None,
                                    'timestamp': str(row["timestamp"]) if pd.notna(row["timestamp"]) else None
                                }
                                yield f"event: crop\ndata: {json.dumps(crop_payload)}\n\n"
            except Exception:
                pass

        # Relinquish CPU execution back to asyncio loop to handle web page requests
        await asyncio.sleep(0.001)

    # Force sweep remaining active tracks and write their final data to the logs
    detection_tracker.purge_old(frame_idx=frame_idx, frames_per_second=frames_per_second, force_flush=True)

    # Drain OCR threads, shut down worker pool, and close file handlers
    drain_pending_ocr(pending_ocr_futures)
    ocr_worker_pool.shutdown(wait=True)
    detection_tracker.flush_all()
    video_capture.release()
    video_writer.release()

    # Transcode final raw output into browser-compatible H.264 format using FFMPEG.
    # -vcodec libx264: H.264 video compression standard compatible with HTML5 players
    # -preset ultrafast: Executes encoding quickly to minimize user wait time
    final_output_video_path = "outputs/results/final_output.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_video_path, "-vcodec", "libx264", "-preset", "ultrafast", final_output_video_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        final_output_video_path = output_video_path

    # Yield final completed events
    final_elapsed = time.time() - start_time
    final_fps = round(processed_frames_count / final_elapsed, 1) if final_elapsed > 0 else 0
    yield f"event: progress\ndata: {json.dumps({'frame_idx': total_frames, 'total_frames': total_frames, 'percent': 100, 'elapsed_str': _format_elapsed(final_elapsed), 'eta': 'Done', 'fps': final_fps})}\n\n"

    yield f"event: complete\ndata: {json.dumps({'video_url': '/outputs/results/final_output.mp4'})}\n\n"


# ─── FastAPI App & Custom Dashboard Serving ───────────────────────

# Initialize FastAPI application instance
app = FastAPI(title="ALPR & Vehicle Classification")

# Mount outputs storage directory as static file path (accessible via web browser)
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Serve custom dashboard index HTML page at root path
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the front-end dashboard UI page."""
    with open("src/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Handle video upload requests
@app.post("/api/upload")
async def upload_video_api(file: UploadFile = File(...)):
    """Receives and saves uploaded video file to the disk uploads directory."""
    os.makedirs("outputs/uploads", exist_ok=True)
    filepath = f"outputs/uploads/{file.filename}"
    with open(filepath, "wb") as f:
        f.write(await file.read())
    return {"filepath": filepath}

# SSE stream process endpoint
@app.get("/api/stream-process")
async def stream_process_api(video_path: str, ocr_engine: str):
    """
    FastAPI streaming endpoint that mounts our process_video_sse generator.
    
    Headers disallow client-side caching and proxy buffering to maintain instant updates.
    """
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",  # Disables nginx proxy buffering so SSE events flow instantly
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        process_video_sse(video_path, ocr_engine),
        media_type="text/event-stream",
        headers=headers
    )

# Serve server log histories
@app.get("/api/logs")
async def get_logs_api():
    """Reads and returns the last 200 lines of standard outputs/errors from server.log."""
    server_log_path = "outputs/logs/server.log"
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

# Serve XLSX Excel report export downloads
@app.get("/api/export")
async def export_api():
    """Generates and downloads the Excel report containing embedded crops and telemetry stats."""
    csv_path = "outputs/logs/detections.csv"
    if not os.path.exists(csv_path):
        return HTMLResponse(content="<h3>No records available to export yet.</h3>", status_code=400)
    
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


# ─── Server Entry Point ──────────────────────────────────────────

if __name__ == "__main__":
    # Start web server hosting FastAPI application on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
