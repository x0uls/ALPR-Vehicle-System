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
    save_ground_truth, load_ground_truth, load_ground_truth_csv,
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



def _parse_csv_row_mapping(row):
    """Extracts filename and ground_truth plate from flexible CSV headers."""
    raw_fname = (
        row.get("filename") or row.get("FilePath") or row.get("file_path") or
        row.get("File") or row.get("image") or row.get("Image") or ""
    ).strip()
    
    fname = os.path.basename(raw_fname) if raw_fname else ""
    
    gt = (
        row.get("ground_truth") or row.get("No. Car Plate") or row.get("Car Plate") or
        row.get("plate_number") or row.get("plate") or row.get("Plate Number") or
        row.get("Plate") or row.get("GT") or ""
    ).strip().upper()
    
    return fname, gt


async def _extract_folder_contents(files: List[UploadFile]):
    """
    Parses uploaded directory files. Validates that the uploaded folder contains:
      - an 'images/' subfolder containing image files
      - a 'csv/' subfolder containing ground truth CSV file(s)
    
    Returns (image_paths, gt_mapping, gt_plates, error_message)
    """
    os.makedirs("outputs/uploads", exist_ok=True)
    
    image_paths = []
    gt_mapping = {}
    gt_plates = []
    
    images_found = False
    csv_found = False
    
    for file in files:
        filename = file.filename if file.filename else ""
        norm_name = filename.replace("\\", "/").lower()
        content = await file.read()
        
        is_in_images = ("/images/" in norm_name or norm_name.startswith("images/"))
        is_in_csv = ("/csv/" in norm_name or norm_name.startswith("csv/"))
        
        is_image_ext = norm_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        is_csv_ext = norm_name.endswith(".csv")
        
        if is_in_images and is_image_ext:
            images_found = True
            base_name = os.path.basename(filename)
            filepath = f"outputs/uploads/{base_name}"
            with open(filepath, "wb") as f:
                f.write(content)
            image_paths.append(filepath)
            
        elif is_in_csv and is_csv_ext:
            csv_found = True
            import csv
            import io
            text = content.decode("utf-8-sig", errors="ignore")
            
            # Save raw ground truth CSV for cer summary
            gt_csv_path = "outputs/logs/ground_truth.csv"
            os.makedirs(os.path.dirname(gt_csv_path), exist_ok=True)
            with open(gt_csv_path, "w", encoding="utf-8", newline="") as out:
                out.write(text)

            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                fname, gt = _parse_csv_row_mapping(row)
                if gt:
                    if fname:
                        gt_mapping[fname] = gt
                        gt_mapping[os.path.splitext(fname)[0]] = gt
                    gt_plates.append(gt)
                    
        elif is_image_ext and "/" not in norm_name:
            images_found = True
            base_name = os.path.basename(filename)
            filepath = f"outputs/uploads/{base_name}"
            with open(filepath, "wb") as f:
                f.write(content)
            image_paths.append(filepath)
            
        elif is_csv_ext and "/" not in norm_name:
            csv_found = True
            import csv
            import io
            text = content.decode("utf-8-sig", errors="ignore")
            gt_csv_path = "outputs/logs/ground_truth.csv"
            os.makedirs(os.path.dirname(gt_csv_path), exist_ok=True)
            with open(gt_csv_path, "w", encoding="utf-8", newline="") as out:
                out.write(text)

            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                fname, gt = _parse_csv_row_mapping(row)
                if gt:
                    if fname:
                        gt_mapping[fname] = gt
                        gt_mapping[os.path.splitext(fname)[0]] = gt
                    gt_plates.append(gt)
    
    if not images_found:
        return None, None, None, "Invalid folder structure: Missing an 'images/' subfolder containing image files."
    if not csv_found:
        return None, None, None, "Invalid folder structure: Missing a 'csv/' subfolder containing ground truth CSV file(s)."
        
    if gt_plates:
        cleaned_gt = list(dict.fromkeys(gt_plates))
        save_ground_truth(cleaned_gt)
        
    return image_paths, gt_mapping, gt_plates, None


@app.post("/api/process-images")
async def process_images_api(files: List[UploadFile] = File(...)):
    try:
        init_log("outputs/logs/detections_easyocr.csv")
        init_log("outputs/logs/detections_pytesseract.csv")
        
        image_paths, gt_mapping, gt_plates, error_msg = await _extract_folder_contents(files)
        if error_msg:
            return JSONResponse({"error": error_msg}, status_code=400)
        
        start_time = time.time()
        results = process_bulk_images(image_paths, easyocr_pool, pytesseract_pool)
        final_elapsed = time.time() - start_time
        
        if not gt_plates:
            gt_plates = load_ground_truth()
            
        easy_csv = "outputs/logs/detections_easyocr.csv"
        tess_csv = "outputs/logs/detections_pytesseract.csv"

        easy_dets = []
        if os.path.exists(easy_csv) and os.path.getsize(easy_csv) > 10:
            try:
                easy_dets = pd.read_csv(easy_csv).to_dict(orient="records")
            except Exception:
                easy_dets = []

        tess_dets = []
        if os.path.exists(tess_csv) and os.path.getsize(tess_csv) > 10:
            try:
                tess_dets = pd.read_csv(tess_csv).to_dict(orient="records")
            except Exception:
                tess_dets = []
        
        for res in results:
            fname = res.get("original_filename", "")
            base_name = os.path.splitext(fname)[0]
            expected = gt_mapping.get(fname) or gt_mapping.get(base_name)
            res["expected_gt"] = expected

        comparison_metrics = compute_dual_model_comparison(easy_dets, tess_dets, gt_plates, easyocr_time=final_elapsed, pytesseract_time=final_elapsed)
        
        return JSONResponse({
            "status": "success",
            "results": results,
            "gt_mapping": gt_mapping,
            "metrics": comparison_metrics,
            "gt_count": len(gt_plates),
            "gt_mapped": len(gt_mapping),
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
        fieldnames = reader.fieldnames or []

        if "ground_truth" in fieldnames and "filename" in fieldnames:
            gt_csv_path = "outputs/logs/ground_truth.csv"
            os.makedirs(os.path.dirname(gt_csv_path), exist_ok=True)
            with open(gt_csv_path, "w", encoding="utf-8", newline="") as out:
                out.write(content)
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                gt = (row.get("ground_truth") or "").strip()
                if gt:
                    plates.append(gt)
        else:
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
    try:
        gt_plates = load_ground_truth()
        easy_csv = "outputs/logs/detections_easyocr.csv"
        tess_csv = "outputs/logs/detections_pytesseract.csv"

        easy_dets = []
        if os.path.exists(easy_csv) and os.path.getsize(easy_csv) > 10:
            try:
                easy_dets = pd.read_csv(easy_csv).to_dict(orient="records")
            except Exception:
                easy_dets = []

        tess_dets = []
        if os.path.exists(tess_csv) and os.path.getsize(tess_csv) > 10:
            try:
                tess_dets = pd.read_csv(tess_csv).to_dict(orient="records")
            except Exception:
                tess_dets = []

        comparison = compute_dual_model_comparison(easy_dets, tess_dets, gt_plates)
        return JSONResponse(comparison)
    except Exception as e:
        import traceback
        print(f"[CER SUMMARY ERROR] {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/benchmark")
async def benchmark_api(files: List[UploadFile] = File(...)):
    """
    Benchmark endpoint: accepts directory upload containing images/ and csv/ subfolders.
    """
    try:
        init_log("outputs/logs/detections_easyocr.csv")
        init_log("outputs/logs/detections_pytesseract.csv")

        image_paths, gt_mapping, gt_plates, error_msg = await _extract_folder_contents(files)
        if error_msg:
            return JSONResponse({"error": error_msg}, status_code=400)

        # Run the full pipeline
        start_time = time.time()
        results = process_bulk_images(image_paths, easyocr_pool, pytesseract_pool)
        total_elapsed = time.time() - start_time

        easy_csv = "outputs/logs/detections_easyocr.csv"
        tess_csv = "outputs/logs/detections_pytesseract.csv"

        easy_dets = []
        if os.path.exists(easy_csv) and os.path.getsize(easy_csv) > 10:
            try:
                easy_dets = pd.read_csv(easy_csv).to_dict(orient="records")
            except Exception:
                easy_dets = []

        tess_dets = []
        if os.path.exists(tess_csv) and os.path.getsize(tess_csv) > 10:
            try:
                tess_dets = pd.read_csv(tess_csv).to_dict(orient="records")
            except Exception:
                tess_dets = []

        if not gt_plates:
            gt_plates = list(gt_mapping.values()) if gt_mapping else load_ground_truth()

        comparison = compute_dual_model_comparison(
            easy_dets, tess_dets, gt_plates,
            easyocr_time=total_elapsed, pytesseract_time=total_elapsed
        )

        per_image_gt = []
        for res in results:
            fname = res.get("original_filename", "")
            expected = gt_mapping.get(fname)
            per_image_gt.append({
                "filename": fname,
                "expected_plate": expected,
                "detections": res.get("detections", [])
            })

        return JSONResponse({
            "status": "success",
            "summary": {
                "total_images": len(image_paths),
                "ground_truth_plates": len(gt_plates),
                "total_detections_easyocr": len(easy_dets),
                "total_detections_pytesseract": len(tess_dets),
                "elapsed": _format_elapsed(total_elapsed)
            },
            "easyocr": comparison.get("easyocr", {}),
            "pytesseract": comparison.get("pytesseract", {}),
            "winner": comparison.get("winner", "Tie"),
            "per_image": per_image_gt,
            "results": results
        })

    except Exception as e:
        import traceback
        print(f"[BENCHMARK ERROR] {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

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


