import csv
import os
import difflib

# Path to the shared CSV storage file where vehicle details are logged
LOG_PATH = "outputs/logs/detections.csv"

# Columns headers used in the detections CSV file
FIELDNAMES = ["track_id", "timestamp", "vehicle_type", "color", "plate_number", "confidence", "snapshot_path", "plate_crop_path"]

def _normalize_plate(plate):
    """
    Standardizes plate text by removing spaces and converting characters to uppercase.
    
    This ensures variations like 'wnd 1234' and 'WND1234' match during lookup.
    """
    if not plate:
        return ""
    return plate.replace(" ", "").upper()

def check_plate_similarity(plate1, plate2):
    """
    Computes a similarity score between two normalized plate strings (from 0.0 to 1.0).
    
    Uses Python's difflib SequenceMatcher to check structural similarity.
    """
    normalized_plate1 = _normalize_plate(plate1)
    normalized_plate2 = _normalize_plate(plate2)
    if not normalized_plate1 or not normalized_plate2:
        return 0.0
    # ratio() returns a score indicating how closely the character sequences align
    return difflib.SequenceMatcher(None, normalized_plate1, normalized_plate2).ratio()

def parse_timestamp_to_seconds(timestamp_string):
    """
    Converts a video timecode string formatted as MM:SS.s into total elapsed seconds (float).
    
    For example: '01:23.4' becomes 83.4 seconds. This makes math comparisons simple.
    """
    try:
        parts = timestamp_string.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
    except Exception:
        pass
    return 0.0


def _safe_float(val, default=0.0):
    """
    Safely casts a variable to float, falling back to a default value if the casting fails.
    
    Prevents program crashes if an OCR engine outputs a null or malformed confidence rating.
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# Global list that stores detection logs temporarily in memory
_log_buffer = []


def init_log():
    """
    Resets the CSV file and deletes cropped images from any previous video processing runs.
    
    Ensures that data from old runs does not pollute or leak into the gallery of the new run.
    """
_log_buffers = {}


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

    import shutil
    for folder in ["outputs/plate_crops/Raw", "outputs/plate_crops/Processed", "outputs/snapshots"]:
        os.makedirs(folder, exist_ok=True)


def _read_all_rows(log_path=LOG_PATH):
    """
    Reads all existing records from the target CSV file.
    """
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_all_rows(rows, log_path=LOG_PATH):
    """
    Writes all log entries to the target CSV file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def log_detection(track_id, vehicle_type, color, plate_number, confidence, snapshot_path, plate_crop_path="", video_timestamp="", log_path=LOG_PATH):
    """
    Appends a new detection row to the specified log buffer.
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


def flush_log(log_path=LOG_PATH):
    """
    Flushes buffered log entries to the specified CSV file on disk.
    """
    global _log_buffers
    buffer = _log_buffers.get(log_path, [])
    if not buffer:
        return

    rows = _read_all_rows(log_path)

    for new_detection_row in buffer:
        updated = False
        for i, row in enumerate(rows):
            if row.get("track_id") == new_detection_row["track_id"]:
                old_conf = _safe_float(row.get("confidence"))
                new_conf = _safe_float(new_detection_row.get("confidence"))
                if new_conf > old_conf:
                    rows[i] = new_detection_row
                updated = True
                break
        
        if not updated and new_detection_row.get("plate_number"):
            for i, row in enumerate(rows):
                if row.get("plate_number") == new_detection_row["plate_number"]:
                    old_conf = _safe_float(row.get("confidence"))
                    new_conf = _safe_float(new_detection_row.get("confidence"))
                    if new_conf > old_conf:
                        rows[i] = new_detection_row
                    updated = True
                    break
                    
        if not updated:
            rows.append(new_detection_row)

    _write_all_rows(rows, log_path)
    _log_buffers[log_path] = []