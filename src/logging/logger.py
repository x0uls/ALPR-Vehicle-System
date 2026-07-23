import csv
import os

LOG_PATH = "outputs/logs/detections.csv"
FIELDNAMES = ["track_id", "timestamp", "vehicle_type", "color", "plate_number", "confidence", "snapshot_path", "plate_crop_path"]
_log_buffers = {}


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def init_log(log_path=LOG_PATH):
    """
    Resets the specified CSV file and clears output crop image directories.
    """
    global _log_buffers
    _log_buffers[log_path] = []
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)

    for folder in ["outputs/plate_crops/Raw", "outputs/plate_crops/Processed", "outputs/snapshots"]:
        os.makedirs(folder, exist_ok=True)


def _read_all_rows(log_path=LOG_PATH):
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_all_rows(rows, log_path=LOG_PATH):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def log_detection(track_id, vehicle_type, color, plate_number, confidence, snapshot_path, plate_crop_path="", video_timestamp="", log_path=LOG_PATH):
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


def flush_log(log_path=LOG_PATH):
    global _log_buffers
    buffer = _log_buffers.get(log_path, [])
    if not buffer:
        return

    rows = _read_all_rows(log_path)

    for new_detection_row in buffer:
        updated = False
        for i, row in enumerate(rows):
            if row.get("track_id") == new_detection_row["track_id"]:
                if _safe_float(new_detection_row.get("confidence")) > _safe_float(row.get("confidence")):
                    rows[i] = new_detection_row
                updated = True
                break
        
        if not updated and new_detection_row.get("plate_number"):
            for i, row in enumerate(rows):
                if row.get("plate_number") == new_detection_row["plate_number"]:
                    if _safe_float(new_detection_row.get("confidence")) > _safe_float(row.get("confidence")):
                        rows[i] = new_detection_row
                    updated = True
                    break
                    
        if not updated:
            rows.append(new_detection_row)

    _write_all_rows(rows, log_path)
    _log_buffers[log_path] = []