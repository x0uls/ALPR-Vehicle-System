"""
Character Error Rate (CER) computation for OCR accuracy evaluation.

CER measures the edit distance between an OCR-predicted plate string and a known
ground truth plate, normalized by the length of the ground truth.

Uses the `jiwer` library which implements CER via the Levenshtein (edit) distance
algorithm under the hood.

--- MATH TUTORIAL FOR YOUR TEACHER ---
CER uses the Levenshtein Distance (a.k.a. Edit Distance) algorithm:

1. Given two strings — the ground truth (what the plate actually says) and the
   prediction (what the OCR engine read) — CER counts the minimum number of
   single-character operations needed to transform the prediction into the truth.

2. The three allowed operations are:
   - Substitution: Replace one character with another (e.g. 'O' → '0')
   - Insertion: Add a missing character (e.g. 'WND123' → 'WND 1234')
   - Deletion: Remove an extra character (e.g. 'WNDD 1234' → 'WND 1234')

3. CER = (Substitutions + Insertions + Deletions) / len(ground_truth)
   - CER of 0.0 means perfect match (no errors at all).
   - CER of 1.0 means every character was wrong.
   - CER can exceed 1.0 if the prediction has more characters than the truth
     (because insertions add to the edit distance beyond the truth length).
"""

import os
import json
import jiwer


# Path to the ground truth storage file
GROUND_TRUTH_PATH = "outputs/logs/ground_truth.json"


def _normalize_plate(plate):
    """
    Normalizes plate text for fair CER comparison.
    
    Strips spaces and converts to uppercase so 'WND 1234' and 'wnd1234' are treated identically.
    """
    if not plate:
        return ""
    return plate.replace(" ", "").upper()


def compute_cer(prediction, ground_truth):
    """
    Computes Character Error Rate between a predicted plate string and ground truth.
    
    Uses jiwer.cer() which computes the Levenshtein-based character error rate internally.
    
    Returns a float between 0.0 (perfect) and potentially >1.0 (more errors than characters).
    Returns None if the ground truth is empty (CER is undefined for empty references).
    """
    pred_normalized = _normalize_plate(prediction)
    truth_normalized = _normalize_plate(ground_truth)
    
    if not truth_normalized:
        return None  # CER undefined when ground truth is empty
    
    if not pred_normalized:
        return 1.0  # Empty prediction against non-empty truth = 100% error
    
    # jiwer.cer() expects string inputs and returns the character error rate as a float
    return jiwer.cer(truth_normalized, pred_normalized)


def find_best_ground_truth_match(prediction, ground_truth_list):
    """
    Finds the ground truth plate that best matches the OCR prediction.
    
    Compares the prediction against every ground truth entry and returns the one
    with the lowest CER (closest match). This handles cases where the ground truth
    list contains many plates and we need to figure out which one the OCR was reading.
    
    Returns a tuple of (best_ground_truth_text, best_cer_score) or (None, None) if
    the ground truth list is empty.
    """
    if not ground_truth_list or not prediction:
        return None, None
    
    pred_normalized = _normalize_plate(prediction)
    if not pred_normalized:
        return None, None
    
    best_match = None
    best_cer = float('inf')
    
    for gt_plate in ground_truth_list:
        gt_normalized = _normalize_plate(gt_plate)
        if not gt_normalized:
            continue
        
        cer = compute_cer(prediction, gt_plate)
        if cer is not None and cer < best_cer:
            best_cer = cer
            best_match = gt_plate
    
    if best_match is None:
        return None, None
    
    return best_match, best_cer


def save_ground_truth(plates_list):
    """
    Saves a list of ground truth plate strings to a JSON file.
    
    Overwrites any existing ground truth data.
    """
    os.makedirs(os.path.dirname(GROUND_TRUTH_PATH), exist_ok=True)
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump({"plates": plates_list}, f, indent=2)


def load_ground_truth():
    """
    Loads the list of ground truth plate strings from the JSON file.
    
    Returns an empty list if the file does not exist.
    """
    if not os.path.exists(GROUND_TRUTH_PATH):
        return []
    try:
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("plates", [])
    except (json.JSONDecodeError, IOError):
        return []


def compute_average_cer(detections_list, ground_truth_list):
    """
    Computes the average CER across all detections that have a ground truth match.
    
    Returns a dictionary with:
    - 'average_cer': The mean CER across all matched detections
    - 'matched_count': How many detections found a ground truth match
    - 'total_detections': Total number of detections evaluated
    - 'per_detection': List of per-detection CER details
    """
    if not ground_truth_list:
        return {
            "average_cer": None,
            "matched_count": 0,
            "total_detections": len(detections_list),
            "per_detection": []
        }
    
    per_detection = []
    total_cer = 0.0
    matched_count = 0
    
    for detection in detections_list:
        plate_text = detection.get("plate_number", "")
        if not plate_text:
            continue
        
        best_gt, best_cer = find_best_ground_truth_match(plate_text, ground_truth_list)
        
        if best_gt is not None:
            matched_count += 1
            total_cer += best_cer
            per_detection.append({
                "plate_number": plate_text,
                "matched_ground_truth": best_gt,
                "cer": round(best_cer, 4)
            })
    
    average_cer = (total_cer / matched_count) if matched_count > 0 else None
    
    return {
        "average_cer": round(average_cer, 4) if average_cer is not None else None,
        "matched_count": matched_count,
        "total_detections": len(detections_list),
        "per_detection": per_detection
    }
