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
    global _log_buffer
    _log_buffer = []
    
    # 1. Reset the CSV detection log file with empty headers
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)

    # 2. Clear old cropped plate images and snapshots to save disk space and clean UI galleries
    import shutil
    for folder in ["outputs/plate_crops/Raw", "outputs/plate_crops/Processed", "outputs/snapshots"]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception:
                pass  # Ignore lock/permission errors on individual files
        os.makedirs(folder, exist_ok=True)


def _read_all_rows():
    """
    Reads all existing records from the detections CSV file.
    
    Returns a list of dictionaries where each dictionary represents a row.
    """
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_all_rows(rows):
    """
    Writes all log entries to the detections CSV file, replacing any existing content.
    """
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def log_detection(track_id, vehicle_type, color, plate_number, confidence, snapshot_path, plate_crop_path="", video_timestamp=""):
    """
    Appends a new detection row to the in-memory buffer.
    
    To avoid writing to disk on every single frame, we buffer logs and auto-flush them every 20 entries.
    """
    _log_buffer.append({
        "track_id": str(track_id),
        "timestamp": video_timestamp,
        "vehicle_type": vehicle_type,
        "color": color,
        "plate_number": plate_number,
        "confidence": confidence,
        "snapshot_path": snapshot_path,
        "plate_crop_path": plate_crop_path,
    })
    # Auto-flush buffer to the CSV file on disk when it hits 20 entries to limit data loss on crash
    if len(_log_buffer) >= 20:
        flush_log()


def flush_log():
    """
    Flushes buffered log entries to the CSV file, merging matching tracks and plate numbers.
    
    Resolves double-detection errors by keeping the entry that achieved the highest confidence.
    """
    global _log_buffer
    if not _log_buffer:
        return

    rows = _read_all_rows()

    for new_detection_row in _log_buffer:
        updated = False
        
        # 1. Deduplicate by track_id first (same tracker identity, confidence improved)
        for i, row in enumerate(rows):
            if row.get("track_id") == new_detection_row["track_id"]:
                old_conf = _safe_float(row.get("confidence"))
                new_conf = _safe_float(new_detection_row.get("confidence"))
                
                # Update the row on disk if the new frame read achieved a higher confidence score
                if new_conf > old_conf:
                    rows[i] = new_detection_row
                updated = True
                break
        
        # 2. Deduplicate by exact plate_number matching (different track ID, same physical plate text)
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

    _write_all_rows(rows)
    _log_buffer = []