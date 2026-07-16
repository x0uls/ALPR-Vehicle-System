import csv
import os
import difflib

# Path to the shared CSV storage file where vehicle details are logged
LOG_PATH = "outputs/logs/detections.csv"

# Columns headers used in the detections CSV file
FIELDNAMES = ["track_id", "timestamp", "vehicle_type", "color", "plate_number", "confidence", "snapshot_path", "plate_crop_path", "canonical_id"]

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

# Caches all unique license plates grouped by their 'canonical_id'
_canonical_plates_cache = {}

def _rebuild_cache_from_rows(rows):
    """
    Scans the CSV rows and rebuilds the global license plate cache.
    
    This allows fast memory lookup of previous reads instead of re-reading files from disk.
    """
    global _canonical_plates_cache
    _canonical_plates_cache = {}
    for row in rows:
        canonical_id = row.get("canonical_id")
        plate = row.get("plate_number")
        if canonical_id and plate:
            normalized_plate = _normalize_plate(plate)
            if normalized_plate:
                # Add normalized plate to the set associated with this canonical ID group
                _canonical_plates_cache.setdefault(canonical_id, set()).add(normalized_plate)


def init_log():
    """
    Resets the CSV file and deletes cropped images from any previous video processing runs.
    
    Ensures that data from old runs does not pollute or leak into the gallery of the new run.
    """
    global _log_buffer, _canonical_plates_cache
    _log_buffer = []
    _canonical_plates_cache = {}
    
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
        "canonical_id": str(track_id),
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
    _rebuild_cache_from_rows(rows)

    for new_detection_row in _log_buffer:
        updated = False
        
        # 1. Deduplicate by track_id first (same tracker identity, confidence improved)
        for i, row in enumerate(rows):
            if row.get("track_id") == new_detection_row["track_id"]:
                old_conf = _safe_float(row.get("confidence"))
                new_conf = _safe_float(new_detection_row.get("confidence"))
                
                # Update the row on disk if the new frame read achieved a higher confidence score
                if new_conf > old_conf:
                    # Inherit the canonical ID to maintain vehicle grouping relationships
                    new_detection_row["canonical_id"] = row.get("canonical_id", row.get("track_id", new_detection_row["track_id"]))
                    rows[i] = new_detection_row
                updated = True
                break
        
        # 2. Deduplicate by plate_number matching (different track ID)
        if not updated and new_detection_row.get("plate_number"):
            new_plate = new_detection_row["plate_number"]
            new_plate_normalized = _normalize_plate(new_plate)
            new_type = new_detection_row["vehicle_type"]
            new_time = parse_timestamp_to_seconds(new_detection_row.get("timestamp", "00:00.0"))
            
            # Phase A: Try exact normalized plate match first against cached history (global, no time/color limit)
            for i, row in enumerate(rows):
                row_type = row.get("vehicle_type")
                if row_type == new_type:
                    row_canonical_id = row.get("canonical_id", row.get("track_id"))
                    cached_plates = _canonical_plates_cache.get(row_canonical_id, set())
                    
                    # If this exact plate was already logged, merge this track into that canonical group
                    if new_plate_normalized in cached_plates:
                        matched_row = rows[i]
                        old_conf = _safe_float(matched_row.get("confidence"))
                        new_conf = _safe_float(new_detection_row.get("confidence"))
                        
                        canonical_id = row_canonical_id
                        if new_conf > old_conf:
                            new_detection_row["canonical_id"] = canonical_id
                            rows[i] = new_detection_row
                        else:
                            matched_row["canonical_id"] = canonical_id
                        
                        _canonical_plates_cache.setdefault(canonical_id, set()).add(new_plate_normalized)
                        updated = True
                        break
            
            # Phase B: Try fuzzy plate match against cached history (within 30s proximity, same vehicle type, no color limit)
            if not updated:
                best_match_idx = -1
                best_match_ratio = 0.0
                best_match_canon_id = None
                
                for i, row in enumerate(rows):
                    row_type = row.get("vehicle_type")
                    row_time = parse_timestamp_to_seconds(row.get("timestamp", "00:00.0"))
                    
                    if row_type == new_type:
                        # 30-second time gap limit. We assume similar plates seen within 30 seconds
                        # are the exact same car (e.g. tracking lost temporarily or plate partially obstructed).
                        if abs(row_time - new_time) <= 30.0:
                            row_canonical_id = row.get("canonical_id", row.get("track_id"))
                            cached_plates = _canonical_plates_cache.get(row_canonical_id, set())
                            
                            # Find the best similarity match in the cache for this canonical ID
                            for cached_plate in cached_plates:
                                ratio = check_plate_similarity(cached_plate, new_plate)
                                # 0.70 similarity threshold allows for 1-2 character mismatches (such as 'B' vs '8')
                                if ratio >= 0.70 and ratio > best_match_ratio:
                                    best_match_ratio = ratio
                                    best_match_idx = i
                                    best_match_canon_id = row_canonical_id
                
                if best_match_idx != -1:
                    matched_row = rows[best_match_idx]
                    old_conf = _safe_float(matched_row.get("confidence"))
                    new_conf = _safe_float(new_detection_row.get("confidence"))
                    
                    canonical_id = best_match_canon_id
                    if new_conf > old_conf:
                        new_detection_row["canonical_id"] = canonical_id
                        rows[best_match_idx] = new_detection_row
                    else:
                        matched_row["canonical_id"] = canonical_id
                    
                    _canonical_plates_cache.setdefault(canonical_id, set()).add(new_plate_normalized)
                    updated = True
                    
        if not updated:
            # Rebuild cache entry for new unique tracks
            canonical_id = new_detection_row.get("canonical_id")
            plate = new_detection_row.get("plate_number")
            if canonical_id and plate:
                normalized_plate = _normalize_plate(plate)
                if normalized_plate:
                    _canonical_plates_cache.setdefault(canonical_id, set()).add(normalized_plate)
            rows.append(new_detection_row)

    _write_all_rows(rows)
    _log_buffer = []