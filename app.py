import glob
import os
import time
import subprocess
import asyncio
import base64
import json

import cv2
import gradio as gr
import pandas as pd
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import FRAME_SKIP, detection_tracker, process_frame, drain_pending_ocr
from src.logging.logger import init_log


LOG_COLUMNS = [
    "track_id",
    "timestamp",
    "vehicle_type",
    "color",
    "plate_number",
    "confidence",
    "snapshot_path",
]


def _empty_log():
    return pd.DataFrame(columns=LOG_COLUMNS)


def _format_elapsed(seconds):
    """Format elapsed seconds into a readable string."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _status_payload(message, progress, elapsed_str="--", button_text="Processing...", button_enabled=False):
    return (
        message,
        progress,
        elapsed_str,
        gr.update(interactive=button_enabled, value=button_text),
    )


# ─── Legacy Gradio Processing Function ───────────────────────────

def process_video(video_path, ocr_engine):
    if not video_path:
        yield (
            None,
            gr.update(visible=False),
            _empty_log(),
            gr.update(),
            *_status_payload("Please upload a video first.", 0, "--", "Process Video", True),
        )
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield (
            None,
            gr.update(visible=False),
            _empty_log(),
            gr.update(),
            *_status_payload("Error: Could not open the uploaded video.", 0, "--", "Process Video", True),
        )
        return

    os.makedirs("outputs/results", exist_ok=True)
    out_path = "outputs/results/processed_output.mp4"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30

    width, height = int(cap.get(3)), int(cap.get(4))
    # Match the pipeline's early downscale — cap output at 1080p
    if width > 1920:
        scale = 1920 / width
        width, height = 1920, int(height * scale)

    # We only write the frames we process, so the output FPS should match the skipped rate
    output_fps = fps / FRAME_SKIP
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, output_fps, (width, height))

    start_time = time.time()

    yield (
        None,
        gr.update(visible=False),
        gr.update(),
        gr.update(),
        *_status_payload("Initializing pipeline and loading models...", 0, "0s"),
    )

    frame_idx = 0
    processed_count = 0

    while True:
        if frame_idx % FRAME_SKIP == 0:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = process_frame(frame, frame_idx, ocr_engine)
            out.write(frame)
            processed_count += 1

            # Update live preview every 2 processed frames (which is every 6 actual frames)
            if processed_count % 2 == 0:
                progress = int((frame_idx / total_frames) * 100) if total_frames else 0
                elapsed = time.time() - start_time
                elapsed_str = _format_elapsed(elapsed)

                # Estimate remaining time
                if progress > 0:
                    eta_seconds = (elapsed / progress) * (100 - progress)
                    eta_str = _format_elapsed(eta_seconds)
                    status = f"Processing frame {frame_idx} / {total_frames} ({progress}%) — ETA: {eta_str}"
                else:
                    status = f"Processing frame {frame_idx}..."

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Downscale live preview frame to save massive WebSocket network bandwidth
                h, w = rgb_frame.shape[:2]
                scale = 640.0 / w if w > 640 else 1.0
                if scale < 1.0:
                    rgb_frame = cv2.resize(rgb_frame, (int(w * scale), int(h * scale)))

                yield (
                    rgb_frame,
                    gr.update(visible=False),
                    gr.update(),
                    gr.update(),
                    *_status_payload(status, progress, elapsed_str),
                )
        else:
            # CPU Optimization: cap.grab() only fetches the frame from demuxer without decompressing it.
            # This avoids wasting CPU decoding skipped 4K frames sequentially.
            ret = cap.grab()
            if not ret:
                break

        frame_idx += 1

    # ── Post-processing: yield status updates so the UI doesn't freeze ──
    elapsed_str = _format_elapsed(time.time() - start_time)
    yield (
        gr.update(),
        gr.update(visible=False),
        gr.update(),
        gr.update(),
        *_status_payload("Finalizing — waiting for remaining OCR results...", 99, elapsed_str),
    )

    # Drain all in-flight OCR futures before generating the final report
    drain_pending_ocr()
    detection_tracker.flush_all()
    cap.release()
    out.release()

    elapsed_str = _format_elapsed(time.time() - start_time)
    yield (
        gr.update(),
        gr.update(visible=False),
        gr.update(),
        gr.update(),
        *_status_payload("Encoding output video (H.264)...", 99, elapsed_str),
    )

    # Convert mp4v to browser-compatible H.264 so Gradio doesn't timeout trying to do it
    # Added -preset ultrafast for massive speedup in CPU-constrained environments (like Colab)
    final_out_path = "outputs/results/final_output.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_path, "-vcodec", "libx264", "-preset", "ultrafast", final_out_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        final_out_path = out_path  # Fallback to the raw mp4v output

    try:
        df = pd.read_csv("outputs/logs/detections.csv")
    except Exception:
        df = _empty_log()

    plate_crops = sorted(glob.glob("outputs/plate_crops/Processed/*.jpg"))
    plate_crops = plate_crops[-30:] if len(plate_crops) > 30 else plate_crops

    detected_count = len(df) if df is not None else 0
    total_elapsed_str = _format_elapsed(time.time() - start_time)
    status = (
        f"Complete — {processed_count} frames processed, "
        f"{detected_count} plate(s) detected in {total_elapsed_str}."
    )
    yield (
        None,
        gr.update(value=final_out_path, visible=True),
        df,
        plate_crops if plate_crops else gr.update(),
        *_status_payload(status, 100, total_elapsed_str, "Process Video", True),
    )


# ─── Server-Sent Events (SSE) Real-Time Generator ────────────────

async def process_video_sse(video_path, ocr_engine):
    # Reset CSV logs and clear crops/snapshots
    init_log()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield f"event: error\ndata: {json.dumps({'error': 'Cannot open video file'})}\n\n"
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30

    width, height = int(cap.get(3)), int(cap.get(4))
    # Cap processing and output size at 1080p
    if width > 1920:
        scale = 1920 / width
        width, height = 1920, int(height * scale)

    os.makedirs("outputs/results", exist_ok=True)
    out_path = "outputs/results/processed_output.mp4"
    output_fps = fps / FRAME_SKIP
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, output_fps, (width, height))

    start_time = time.time()
    frame_idx = 0
    processed_count = 0

    sent_logs = set()
    sent_crops = set()

    while True:
        if frame_idx % FRAME_SKIP == 0:
            ret, frame = cap.read()
            if not ret:
                break

            # Run ALPR pipeline frame processing
            frame = process_frame(frame, frame_idx, ocr_engine)
            out.write(frame)
            processed_count += 1

            # Send state updates every 2 processed frames (which is 6 actual frames)
            if processed_count % 2 == 0:
                elapsed = time.time() - start_time
                elapsed_str = _format_elapsed(elapsed)
                progress_val = int((frame_idx / total_frames) * 100) if total_frames else 0

                eta_str = "--"
                if progress_val > 0:
                    eta_seconds = (elapsed / progress_val) * (100 - progress_val)
                    eta_str = _format_elapsed(eta_seconds)

                fps_val = round(processed_count / elapsed, 1) if elapsed > 0 else 0

                # Yield progress stats
                progress_data = {
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "percent": progress_val,
                    "elapsed_str": elapsed_str,
                    "eta": eta_str,
                    "fps": fps_val
                }
                yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"

                # Downscale & encode frame base64 to stream preview to canvas
                h_img, w_img = frame.shape[:2]
                scale_img = 480.0 / w_img if w_img > 480 else 1.0
                preview_frame = frame
                if scale_img < 1.0:
                    preview_frame = cv2.resize(frame, (int(w_img * scale_img), int(h_img * scale_img)))

                _, buffer = cv2.imencode('.jpg', preview_frame)
                img_b64 = base64.b64encode(buffer).decode('utf-8')
                yield f"event: frame\ndata: {json.dumps({'image': img_b64})}\n\n"

                # Check and stream new CSV records
                try:
                    df = pd.read_csv("outputs/logs/detections.csv")
                    for _, row in df.iterrows():
                        track_key = f"{row['track_id']}_{row['confidence']}"
                        if track_key not in sent_logs:
                            sent_logs.add(track_key)
                            yield f"event: log\ndata: {json.dumps(row.to_dict())}\n\n"
                except Exception:
                    pass

                # Check and stream newly cropped plate images
                try:
                    crop_files = glob.glob("outputs/plate_crops/Processed/*.jpg")
                    for filepath in crop_files:
                        filename = os.path.basename(filepath)
                        if filename not in sent_crops:
                            sent_crops.add(filename)
                            
                            text_read = "Plate"
                            try:
                                track_id = int(filename.split("track")[1].split("_")[0])
                                match = df[df["track_id"] == track_id]
                                if not match.empty:
                                    text_read = match.iloc[0]["plate_number"]
                            except Exception:
                                pass
                            
                            yield f"event: crop\ndata: {json.dumps({'filename': filename, 'url': f'/outputs/plate_crops/Processed/{filename}', 'text': text_read})}\n\n"
                except Exception:
                    pass
        else:
            ret = cap.grab()
            if not ret:
                break

        frame_idx += 1
        # Relinquish CPU execution control back to asyncio event loop
        await asyncio.sleep(0.001)

    # Drain OCR threads
    drain_pending_ocr()
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

    yield f"event: complete\ndata: {json.dumps({'video_url': '/outputs/results/final_output.mp4'})}\n\n"


# ─── Legacy Gradio Theme & Layout ────────────────────────────────

theme = gr.themes.Default(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.status-bar {
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    border: 1px solid var(--border-color-primary);
}
.elapsed-display {
    font-size: 24px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    text-align: center;
    padding: 8px;
}
.header-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
footer { display: none !important; }
"""

with gr.Blocks(title="ALPR & Vehicle Classification", theme=theme, css=CSS) as demo:
    gr.Markdown("# Automated License Plate Recognition & Vehicle Classification")
    gr.Markdown("Upload a video to parse license plates using the YOLO & OCR engine.")

    with gr.Group():
        gr.Markdown("### Input")
        with gr.Row(equal_height=True):
            video_input = gr.Video(label="Upload Video")
            with gr.Column(scale=0, min_width=220):
                ocr_engine_input = gr.Dropdown(
                    ["EasyOCR", "PyTesseract"],
                    value="EasyOCR",
                    label="OCR Engine",
                )
                process_btn = gr.Button("Process Video", variant="primary", size="lg")

    with gr.Group():
        gr.Markdown("### Processing Status")
        with gr.Row():
            with gr.Column(scale=4):
                status_box = gr.Markdown("Ready.", elem_classes=["status-bar"])
                progress_bar = gr.Slider(label="Progress", minimum=0, maximum=100, value=0, step=1, interactive=False)
            with gr.Column(scale=1, min_width=120):
                elapsed_display = gr.Markdown("--", elem_classes=["elapsed-display"])
                gr.Markdown("<center style='opacity:0.6; font-size:12px;'>Time Elapsed</center>")

    with gr.Group():
        gr.Markdown("### Output")
        live_stream = gr.Image(label="Live Detection Stream")
        final_video = gr.Video(label="Processed Video", visible=False)

    log_table = gr.Dataframe(label="Detected Vehicles & Plates")
    plate_gallery = gr.Gallery(label="Processed plate images sent to OCR", columns=6, height="auto")

    process_btn.click(
        fn=process_video,
        inputs=[video_input, ocr_engine_input],
        outputs=[
            live_stream,
            final_video,
            log_table,
            plate_gallery,
            status_box,
            progress_bar,
            elapsed_display,
            process_btn,
        ],
    )


# ─── FastAPI Mounting & Custom Dashboard Serving ──────────────────

# Expose Gradio's FastAPI app instance
app = demo.app

# Mount output storage locally
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Remove Gradio's default root "/" mapping
for route in list(app.routes):
    if route.path == "/":
        app.routes.remove(route)

# Register custom dashboard serving at root "/"
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

# Launch the Gradio Blocks app (this triggers the sharing tunnel)
demo.launch()
