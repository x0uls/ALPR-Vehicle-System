import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import process_bulk_images
from src.logging.logger import init_log
from src.metrics.cer import (
    save_ground_truth, load_ground_truth,
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


# ─── Thread Pools ──────────────────────────────────────────────────
# Create global thread pools for OCR to avoid reinitializing them per request
easyocr_pool = ThreadPoolExecutor(max_workers=2)
pytesseract_pool = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 2, 6))



# ─── FastAPI App & Custom Dashboard Serving ───────────────────────

app = FastAPI(title="ALPR & Vehicle Classification - Dual Model Comparison")

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("src/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())



@app.post("/api/process-images")
async def process_images_api(files: List[UploadFile] = File(...)):
    try:
        os.makedirs("outputs/uploads", exist_ok=True)
        init_log("outputs/logs/detections_easyocr.csv")
        init_log("outputs/logs/detections_pytesseract.csv")
        
        image_paths = []
        for file in files:
            filename = file.filename if file.filename else f"upload_{int(time.time())}.jpg"
            filepath = f"outputs/uploads/{filename}"
            with open(filepath, "wb") as f:
                content = await file.read()
                f.write(content)
            image_paths.append(filepath)
        
        # Process the bulk images
        start_time = time.time()
        results = process_bulk_images(image_paths, easyocr_pool, pytesseract_pool)
        final_elapsed = time.time() - start_time
        
        gt_plates = load_ground_truth()
        easy_dets = pd.read_csv("outputs/logs/detections_easyocr.csv").to_dict(orient="records") if os.path.exists("outputs/logs/detections_easyocr.csv") else []
        tess_dets = pd.read_csv("outputs/logs/detections_pytesseract.csv").to_dict(orient="records") if os.path.exists("outputs/logs/detections_pytesseract.csv") else []
        
        comparison_metrics = compute_dual_model_comparison(easy_dets, tess_dets, gt_plates, easyocr_time=final_elapsed, pytesseract_time=final_elapsed)
        
        return JSONResponse({
            "status": "success",
            "results": results,
            "metrics": comparison_metrics,
            "elapsed": _format_elapsed(final_elapsed)
        })
        
    except Exception as e:
        import traceback
        print(f"[PROCESS IMAGES ERROR] {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


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


