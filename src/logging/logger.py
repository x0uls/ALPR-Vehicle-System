import csv
import os
from datetime import datetime

LOG_PATH = "outputs/logs/detections.csv"
FIELDNAMES = ["track_id", "timestamp", "vehicle_type", "color", "plate_number", "confidence", "snapshot_path"]

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


def log_detection(track_id, vehicle_type, color, plate_number, confidence, snapshot_path):
    """Buffer a detection row. Flushes to disk automatically every 20 entries.
    
    Call flush_log() at end-of-video to ensure all buffered entries are written.
    Deduplication by track_id happens at flush time.
    """
    _log_buffer.append({
        "track_id": str(track_id),
        "timestamp": datetime.now().isoformat(),
        "vehicle_type": vehicle_type,
        "color": color,
        "plate_number": plate_number,
        "confidence": confidence,
        "snapshot_path": snapshot_path,
    })
    # Auto-flush every 20 entries to limit data loss on crash
    if len(_log_buffer) >= 20:
        flush_log()


def flush_log():
    """Flush buffered detections to CSV, deduplicating by plate_number (keeps highest confidence)
    or by track_id (keeps latest)."""
    global _log_buffer
    if not _log_buffer:
        return

    rows = _read_all_rows()

    for new_row in _log_buffer:
        updated = False
        new_plate = new_row["plate_number"].replace(" ", "").upper()
        
        # 1. Deduplicate by plate number first
        for i, row in enumerate(rows):
            row_plate = row.get("plate_number", "").replace(" ", "").upper()
            if row_plate == new_plate:
                try:
                    old_conf = float(row.get("confidence", 0.0))
                    new_conf = float(new_row["confidence"])
                except ValueError:
                    old_conf, new_conf = 0.0, 0.0
                
                # Keep the entry with the higher confidence reading
                if new_conf > old_conf:
                    rows[i] = new_row
                updated = True
                break
                
        # 2. Fallback: Deduplicate by track_id
        if not updated:
            for i, row in enumerate(rows):
                if row.get("track_id") == new_row["track_id"]:
                    rows[i] = new_row
                    updated = True
                    break
                    
        if not updated:
            rows.append(new_row)

    _write_all_rows(rows)
    _log_buffer = []