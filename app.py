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
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m > 0 else f"{s}s"


# ─── Thread Pools & App Setup ──────────────────────────────────────
easyocr_pool = ThreadPoolExecutor(max_workers=2)
pytesseract_pool = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 2, 6))

app = FastAPI(title="ALPR Dual-Model Benchmarking Platform")

os.makedirs("outputs", exist_ok=True)
os.makedirs("src/static", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="src/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("src/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def _parse_csv_row_mapping(row):
    fname = (row.get("filename") or "").strip()
    gt = (row.get("ground_truth") or "").strip().upper()
    return os.path.basename(fname) if fname else "", gt


async def _extract_folder_contents(files: List[UploadFile]):
    image_inputs, gt_mapping, gt_plates = [], {}, []
    images_found, csv_found = False, False

    for file in files:
        filename = file.filename or ""
        norm_name = filename.replace("\\", "/").lower()
        content = await file.read()

        is_in_images = ("/images/" in norm_name or norm_name.startswith("images/"))
        is_in_csv = ("/csv/" in norm_name or norm_name.startswith("csv/"))
        is_image_ext = norm_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        is_csv_ext = norm_name.endswith(".csv")

        if (is_in_images and is_image_ext) or (is_image_ext and "/" not in norm_name):
            images_found = True
            image_inputs.append((os.path.basename(filename), content))

        elif (is_in_csv and is_csv_ext) or (is_csv_ext and "/" not in norm_name):
            csv_found = True
            text = content.decode("utf-8-sig", errors="ignore")
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
        save_ground_truth(list(dict.fromkeys(gt_plates)))

    return image_inputs, gt_mapping, gt_plates, None


@app.post("/api/process-images")
async def process_images_api(files: List[UploadFile] = File(...)):
    try:
        image_inputs, gt_mapping, gt_plates, error_msg = await _extract_folder_contents(files)
        if error_msg:
            return JSONResponse({"error": error_msg}, status_code=400)

        start_time = time.time()
        pipeline_out = process_bulk_images(image_inputs, easyocr_pool, pytesseract_pool)
        final_elapsed = time.time() - start_time

        results = pipeline_out.get("results", []) if isinstance(pipeline_out, dict) else pipeline_out
        discarded_stats = pipeline_out.get("discarded_stats", {}) if isinstance(pipeline_out, dict) else {"total_discarded": 0, "no_vehicle_count": 0, "no_plate_count": 0, "discarded_files": []}

        if not gt_plates:
            gt_plates = load_ground_truth()

        easy_dets, tess_dets = [], []
        for res in results:
            fname = res.get("original_filename", "")
            expected = gt_mapping.get(fname) or gt_mapping.get(os.path.splitext(fname)[0])
            res["expected_gt"] = expected

            for det in res.get("detections", []):
                e = det.get("easyocr", {})
                if e.get("plate_text"):
                    easy_dets.append({"track_id": det.get("det_id"), "plate_number": e.get("plate_text"), "confidence": e.get("conf", 0.0), "snapshot_path": e.get("snapshot_url"), "plate_crop_path": e.get("crop_url")})
                t = det.get("pytesseract", {})
                if t.get("plate_text"):
                    tess_dets.append({"track_id": det.get("det_id"), "plate_number": t.get("plate_text"), "confidence": t.get("conf", 0.0), "snapshot_path": t.get("snapshot_url"), "plate_crop_path": t.get("crop_url")})

        metrics = compute_dual_model_comparison(easy_dets, tess_dets, gt_plates, easyocr_time=final_elapsed, pytesseract_time=final_elapsed)

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
    return JSONResponse({"plates": load_ground_truth()})


@app.post("/api/ground-truth")
async def set_ground_truth(request: Request):
    body = await request.json()
    plates = body.get("plates", [])
    cleaned = list(dict.fromkeys(p.strip() for p in plates if p.strip()))
    save_ground_truth(cleaned)
    return JSONResponse({"status": "ok", "count": len(cleaned), "plates": cleaned})


@app.get("/api/cer-summary")
async def cer_summary_api():
    try:
        return JSONResponse(compute_dual_model_comparison([], [], load_ground_truth()))
    except Exception as e:
        print(f"[CER SUMMARY ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app)
