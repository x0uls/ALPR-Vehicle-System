import csv
import os
import difflib
from datetime import datetime

LOG_PATH = "outputs/logs/detections.csv"
FIELDNAMES = ["track_id", "timestamp", "vehicle_type", "color", "plate_number", "confidence", "snapshot_path", "plate_crop_path", "canonical_id"]

def _normalize_plate(plate):
    if not plate:
        return ""
    return plate.replace(" ", "").upper()

def check_plate_similarity(plate1, plate2):
    p1 = _normalize_plate(plate1)
    p2 = _normalize_plate(plate2)
    if not p1 or not p2:
        return 0.0
    return difflib.SequenceMatcher(None, p1, p2).ratio()

def parse_timestamp_to_seconds(ts_str):
    """Parse MM:SS.s into total seconds float."""
    try:
        parts = ts_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
    except Exception:
        pass
    return 0.0

_log_buffer = []


def init_log():
    """Reset the detection log and clear previous run outputs (crops, snapshots)."""
    global _log_buffer
    _log_buffer = []
    
    # 1. Reset the CSV detection log
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)

    # 2. Clear old image files so they don't leak into the gallery of the new run
    import shutil
    for folder in ["outputs/plate_crops/Raw", "outputs/plate_crops/Processed", "outputs/snapshots"]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception:
                pass  # Ignore lock/permission errors on individual files
        os.makedirs(folder, exist_ok=True)


def _read_all_rows():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_all_rows(rows):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def log_detection(track_id, vehicle_type, color, plate_number, confidence, snapshot_path, plate_crop_path="", video_timestamp=""):
    """Buffer a detection row. Flushes to disk automatically every 20 entries.
    
    Call flush_log() at end-of-video to ensure all buffered entries are written.
    Deduplication by track_id happens at flush time.

    Args:
        video_timestamp: The video position string (e.g. '01:23.4') computed from frame_idx / fps.
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
        "canonical_id": str(track_id),
    })
    # Auto-flush every 20 entries to limit data loss on crash
    if len(_log_buffer) >= 20:
        flush_log()


def flush_log():
    """Flush buffered detections to CSV, deduplicating by track_id AND fuzzy plate_number matching (keeps highest confidence)."""
    global _log_buffer
    if not _log_buffer:
        return

    rows = _read_all_rows()

    for new_row in _log_buffer:
        updated = False
        
        # 1. Deduplicate by track_id first (same tracker identity, confidence improved)
        for i, row in enumerate(rows):
            if row.get("track_id") == new_row["track_id"]:
                try:
                    old_conf = float(row.get("confidence", 0.0))
                    new_conf = float(new_row["confidence"])
                except ValueError:
                    old_conf, new_conf = 0.0, 0.0
                
                # Keep the entry with the higher confidence reading
                if new_conf > old_conf:
                    new_row["canonical_id"] = row.get("canonical_id", row.get("track_id", new_row["track_id"]))
                    rows[i] = new_row
                updated = True
                break
        
        # 2. Deduplicate by fuzzy plate_number matching (different track ID, similar plate, same vehicle type and color, close in time)
        if not updated and new_row.get("plate_number"):
            best_match_idx = -1
            best_match_ratio = 0.0
            
            new_plate = new_row["plate_number"]
            new_type = new_row["vehicle_type"]
            new_color = (new_row.get("color") or "").strip().lower()
            new_time = parse_timestamp_to_seconds(new_row.get("timestamp", "00:00.0"))
            
            for i, row in enumerate(rows):
                row_plate = row.get("plate_number")
                row_type = row.get("vehicle_type")
                row_color = (row.get("color") or "").strip().lower()
                row_time = parse_timestamp_to_seconds(row.get("timestamp", "00:00.0"))
                
                if row_plate and row_type == new_type and row_color == new_color:
                    if abs(row_time - new_time) <= 15.0:
                        ratio = check_plate_similarity(row_plate, new_plate)
                        if ratio >= 0.70 and ratio > best_match_ratio:
                            best_match_ratio = ratio
                            best_match_idx = i
            
            if best_match_idx != -1:
                matched_row = rows[best_match_idx]
                try:
                    old_conf = float(matched_row.get("confidence", 0.0))
                    new_conf = float(new_row["confidence"])
                except ValueError:
                    old_conf, new_conf = 0.0, 0.0
                
                # Keep the canonical_id of the existing row we matched with
                canonical_id = matched_row.get("canonical_id", matched_row.get("track_id", new_row["track_id"]))
                
                if new_conf > old_conf:
                    new_row["canonical_id"] = canonical_id
                    rows[best_match_idx] = new_row
                else:
                    matched_row["canonical_id"] = canonical_id
                
                updated = True
                    
        if not updated:
            rows.append(new_row)

    _write_all_rows(rows)
    _log_buffer = []