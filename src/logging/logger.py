import os
import pandas as pd

LOG_PATH = "outputs/logs/detections.csv"
FIELDNAMES = ["track_id", "timestamp", "vehicle_type", "color", "plate_number", "confidence", "snapshot_path", "plate_crop_path"]

_log_buffer = []


def init_log():
    """Reset the detection log and clear previous run outputs (crops, snapshots)."""
    global _log_buffer
    _log_buffer = []
    
    # 1. Reset the CSV detection log
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    pd.DataFrame(columns=FIELDNAMES).to_csv(LOG_PATH, index=False)

    # 2. Clear old image files so they don't leak into the gallery of the new run
    import shutil
    for folder in ["outputs/plate_crops/Raw", "outputs/plate_crops/Processed", "outputs/snapshots"]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception:
                pass  # Ignore lock/permission errors on individual files
        os.makedirs(folder, exist_ok=True)


def log_detection(track_id, vehicle_type, color, plate_number, confidence, snapshot_path, plate_crop_path="", video_timestamp=""):
    """Buffer a detection row. Flushes to disk automatically every 20 entries."""
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
    # Auto-flush every 20 entries to limit data loss on crash
    if len(_log_buffer) >= 20:
        flush_log()


def flush_log():
    """Flush buffered detections to CSV, deduplicating using pandas DataFrame operations."""
    global _log_buffer
    if not _log_buffer:
        return

    # Read existing log
    if os.path.exists(LOG_PATH):
        try:
            df = pd.read_csv(LOG_PATH)
        except Exception:
            df = pd.DataFrame(columns=FIELDNAMES)
    else:
        df = pd.DataFrame(columns=FIELDNAMES)

    # Concatenate with new entries
    df_new = pd.DataFrame(_log_buffer)
    df = pd.concat([df, df_new], ignore_index=True)

    # Coerce confidence to float for comparison
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    
    # Generate clean normalized plate field for duplicate resolution
    df["plate_clean"] = df["plate_number"].astype(str).str.replace(" ", "").str.upper()

    # Sort descending by confidence so drop_duplicates keeps the highest confidence reads
    df = df.sort_values(by="confidence", ascending=False)
    
    # 1. Deduplicate by plate number
    df = df.drop_duplicates(subset=["plate_clean"], keep="first")
    
    # 2. Deduplicate by track_id
    df = df.drop_duplicates(subset=["track_id"], keep="first")

    # Sort by track ID index for neatness
    df["track_id_int"] = pd.to_numeric(df["track_id"], errors="coerce").fillna(999999)
    df = df.sort_values(by="track_id_int").drop(columns=["track_id_int", "plate_clean"])

    # Ensure correct column order and save
    df = df[FIELDNAMES]
    df.to_csv(LOG_PATH, index=False)
    
    _log_buffer = []