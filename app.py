import os
import time
import csv
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List

import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import process_bulk_images
from src.metrics.cer import (
    save_ground_truth, load_ground_truth,
    compute_dual_model_comparison
)


def _format_elapsed(seconds):
    """Formats a duration in seconds into a human-readable string like '2m 15s' or '8s'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m > 0 else f"{s}s"


# ─── Thread Pools & App Setup ──────────────────────────────────────
# EasyOCR is GPU-bound and has a global lock, so only 2 workers
easyocr_pool = ThreadPoolExecutor(max_workers=2)
# PyTesseract is CPU-bound, so scale workers to available cores (capped at 6)
pytesseract_pool = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 2, 6))

app = FastAPI(title="ALPR Dual-Model Benchmarking Platform")

# Create output directory for any saved files
os.makedirs("outputs", exist_ok=True)
os.makedirs("src/static", exist_ok=True)
# Mount static file directories so the browser can access them via URL paths
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="src/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the main HTML dashboard page."""
    with open("src/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def _parse_csv_row_mapping(row):
    """Extracts (filename, ground_truth) from a single CSV row.
    Strips whitespace and uppercases the ground truth text."""
    fname = (row.get("filename") or "").strip()
    gt = (row.get("ground_truth") or "").strip().upper()
    return os.path.basename(fname) if fname else "", gt


async def _extract_folder_contents(files: List[UploadFile]):
    """Separates uploaded files into images and CSV ground truth.

    Walks through all uploaded files and classifies them based on:
    - File extension (.jpg/.png = image, .csv = ground truth)
    - Folder structure (files in /images/ or /csv/ subfolders)

    Returns:
        image_inputs: List of (filename, raw_bytes) tuples for processing
        gt_mapping: Dict mapping filename → expected plate text
        gt_plates: List of all unique ground truth plate strings
        error_msg: Error string if the folder structure is invalid, else None
    """
    image_inputs, gt_mapping, gt_plates = [], {}, []
    images_found, csv_found = False, False

    for file in files:
        filename = file.filename or ""
        # Normalize path separators for cross-platform compatibility
        norm_name = filename.replace("\\", "/").lower()
        # Read the entire file content into memory
        content = await file.read()

        # Check which subfolder this file belongs to
        is_in_images = ("/images/" in norm_name or norm_name.startswith("images/"))
        is_in_csv = ("/csv/" in norm_name or norm_name.startswith("csv/"))
        is_image_ext = norm_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        is_csv_ext = norm_name.endswith(".csv")

        # Collect image files — either from /images/ subfolder or root-level loose files
        if (is_in_images and is_image_ext) or (is_image_ext and "/" not in norm_name):
            images_found = True
            image_inputs.append((os.path.basename(filename), content))

        # Parse CSV ground truth files — maps filenames to expected plate text
        elif (is_in_csv and is_csv_ext) or (is_csv_ext and "/" not in norm_name):
            csv_found = True
            text = content.decode("utf-8-sig", errors="ignore")
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                fname, gt = _parse_csv_row_mapping(row)
                if gt:
                    if fname:
                        # Store mapping for both "car1.jpg" and "car1" (without extension)
                        gt_mapping[fname] = gt
                        gt_mapping[os.path.splitext(fname)[0]] = gt
                    gt_plates.append(gt)

    # Validate that both images and CSV were found
    if not images_found:
        return None, None, None, "Invalid folder structure: Missing an 'images/' subfolder containing image files."
    if not csv_found:
        return None, None, None, "Invalid folder structure: Missing a 'csv/' subfolder containing ground truth CSV file(s)."

    # Persist ground truth plates for later use (e.g. CER summary endpoint)
    if gt_plates:
        save_ground_truth(list(dict.fromkeys(gt_plates)))

    return image_inputs, gt_mapping, gt_plates, None


@app.post("/api/process-images")
async def process_images_api(files: List[UploadFile] = File(...)):
    """Main API endpoint — receives uploaded dataset, runs the full processing pipeline,
    computes benchmarking metrics, and returns everything as JSON."""
    try:
        # Step 1: Separate images from CSV ground truth
        image_inputs, gt_mapping, gt_plates, error_msg = await _extract_folder_contents(files)
        if error_msg:
            return JSONResponse({"error": error_msg}, status_code=400)

        # Step 2: Run the core pipeline (vehicle detection → plate detection → dual OCR)
        start_time = time.time()
        pipeline_out = process_bulk_images(image_inputs, easyocr_pool, pytesseract_pool)
        final_elapsed = time.time() - start_time

        # Extract results and discard stats from pipeline output
        results = pipeline_out.get("results", []) if isinstance(pipeline_out, dict) else pipeline_out
        discarded_stats = pipeline_out.get("discarded_stats", {}) if isinstance(pipeline_out, dict) else {"total_discarded": 0, "no_vehicle_count": 0, "no_plate_count": 0, "discarded_files": []}

        # Fall back to previously saved ground truth if none was uploaded
        if not gt_plates:
            gt_plates = load_ground_truth()

        # Step 3: Build detection arrays for metrics computation
        # Attach the expected ground truth to each result and collect per-engine detections
        easy_dets, tess_dets = [], []
        for res in results:
            fname = res.get("original_filename", "")
            # Look up ground truth by filename (with and without extension)
            expected = gt_mapping.get(fname) or gt_mapping.get(os.path.splitext(fname)[0])
            res["expected_gt"] = expected

            for det in res.get("detections", []):
                e = det.get("easyocr", {})
                if e.get("plate_text"):
                    easy_dets.append({"track_id": det.get("det_id"), "plate_number": e.get("plate_text"), "confidence": e.get("conf", 0.0), "snapshot_path": e.get("snapshot_url"), "plate_crop_path": e.get("crop_url"), "matched_ground_truth": expected, "file_name": fname})
                t = det.get("pytesseract", {})
                if t.get("plate_text"):
                    tess_dets.append({"track_id": det.get("det_id"), "plate_number": t.get("plate_text"), "confidence": t.get("conf", 0.0), "snapshot_path": t.get("snapshot_url"), "plate_crop_path": t.get("crop_url"), "matched_ground_truth": expected, "file_name": fname})

        # Step 4: Compute comparison metrics (CER, exact match rate, CRR, chart)
        metrics = compute_dual_model_comparison(easy_dets, tess_dets, gt_plates, easyocr_time=final_elapsed, pytesseract_time=final_elapsed)

        # Step 5: Return everything to the frontend as JSON
        return JSONResponse({
            "status": "success",
            "results": results,
            "discarded_stats": discarded_stats,
            "gt_mapping": gt_mapping,
            "metrics": metrics,
            "gt_count": len(gt_plates),
            "gt_mapped": len(gt_mapping),
            "elapsed": _format_elapsed(final_elapsed)
        })

    except Exception as e:
        print(f"[PROCESS IMAGES ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/ground-truth")
async def get_ground_truth():
    """Returns the currently saved list of ground truth plate strings."""
    return JSONResponse({"plates": load_ground_truth()})


@app.post("/api/ground-truth")
async def set_ground_truth(request: Request):
    """Manually sets ground truth plates from a JSON body with a 'plates' array."""
    body = await request.json()
    plates = body.get("plates", [])
    # Deduplicate and clean whitespace
    cleaned = list(dict.fromkeys(p.strip() for p in plates if p.strip()))
    save_ground_truth(cleaned)
    return JSONResponse({"status": "ok", "count": len(cleaned), "plates": cleaned})


@app.get("/api/cer-summary")
async def cer_summary_api():
    """Returns a metrics summary without running any new processing — uses previously saved data."""
    try:
        return JSONResponse(compute_dual_model_comparison([], [], load_ground_truth()))
    except Exception as e:
        print(f"[CER SUMMARY ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app)
