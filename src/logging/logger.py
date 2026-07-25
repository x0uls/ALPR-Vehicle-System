import csv
import os

# Standard CSV file structure field names
LOG_PATH = "outputs/logs/detections_easyocr.csv"
FIELDNAMES = ["track_id", "timestamp", "vehicle_type", "color", "plate_number", "confidence", "snapshot_path", "plate_crop_path"]
_log_buffers = {}


def _safe_float(val, default=0.0):
    """Safely converts string or numerical inputs to a float with a default fallback."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def init_log(log_path=LOG_PATH):
    """
    Resets the specified CSV file and initializes output crop image directories.
    
    Creates CSV header rows and ensures output directories for processed crops and snapshots exist.
    """
    _log_buffers[log_path] = []
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)

    for folder in ["outputs/plate_crops/Processed", "outputs/snapshots"]:
        os.makedirs(folder, exist_ok=True)


def _read_all_rows(log_path=LOG_PATH):
    """Reads all existing records from the specified CSV file into a list of dictionaries."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_all_rows(rows, log_path=LOG_PATH):
    """Overwrites the CSV file with the updated list of detection records."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def log_detection(track_id, vehicle_type, color, plate_number, confidence, snapshot_path, plate_crop_path="", video_timestamp="", log_path=LOG_PATH):
    """
    Buffers a new vehicle detection event in memory.
    
    Flushes records to CSV file when buffer reaches 20 items to minimize disk I/O latency.
    """
    if log_path not in _log_buffers:
        _log_buffers[log_path] = []

    _log_buffers[log_path].append({
        "track_id": str(track_id),
        "timestamp": video_timestamp,
        "vehicle_type": vehicle_type,
        "color": color,
        "plate_number": plate_number,
        "confidence": confidence,
        "snapshot_path": snapshot_path,
        "plate_crop_path": plate_crop_path,
    })
    if len(_log_buffers[log_path]) >= 20:
        flush_log(log_path)


def _normalize_plate_str(p):
    """Normalizes plate string by stripping spaces and converting to uppercase for deduplication."""
    return p.replace(" ", "").upper() if p else ""


def flush_log(log_path=LOG_PATH):
    """
    Flushes buffered detection logs from memory to disk.
    
    Applies deduplication: if a track or plate number already exists in the log, updates the record
    only if the new detection achieved a higher OCR confidence score.
    """
    buffer = _log_buffers.get(log_path, [])
    if not buffer:
        return

    rows = _read_all_rows(log_path)

    for new_detection_row in buffer:
        updated = False
        # Rule 1: Match by unique track_id
        for i, row in enumerate(rows):
            if row.get("track_id") == new_detection_row["track_id"]:
                if _safe_float(new_detection_row.get("confidence")) > _safe_float(row.get("confidence")):
                    rows[i] = new_detection_row
                updated = True
                break
        
        # Rule 2: Deduplicate identical plate text reads by keeping the highest confidence entry
        if not updated and new_detection_row.get("plate_number"):
            new_norm = _normalize_plate_str(new_detection_row["plate_number"])
            if new_norm:
                for i, row in enumerate(rows):
                    if _normalize_plate_str(row.get("plate_number")) == new_norm:
                        if _safe_float(new_detection_row.get("confidence")) > _safe_float(row.get("confidence")):
                            rows[i] = new_detection_row
                        updated = True
                        break
                    
        if not updated:
            rows.append(new_detection_row)

    _write_all_rows(rows, log_path)
    _log_buffers[log_path] = []