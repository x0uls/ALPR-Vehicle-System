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


def compute_edit_distance(prediction, ground_truth):
    """
    Computes the raw Levenshtein edit distance (total character insertions, deletions, substitutions)
    between prediction and ground truth.
    """
    pred_norm = _normalize_plate(prediction)
    truth_norm = _normalize_plate(ground_truth)
    if not truth_norm:
        return 0
    if not pred_norm:
        return len(truth_norm)
    output = jiwer.process_characters(truth_norm, pred_norm)
    return output.substitutions + output.deletions + output.insertions


def compute_comprehensive_metrics(detections_list, ground_truth_list, execution_time_seconds=0.0):
    """
    Computes comprehensive Ground Truth evaluation metrics for an OCR model's detections.
    """
    if not detections_list:
        return {
            "average_cer": None,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "gt_recall": 0.0,
            "precision": 0.0,
            "total_edit_distance": 0,
            "average_confidence": 0.0,
            "correct_confidence": 0.0,
            "incorrect_confidence": 0.0,
            "total_detections": 0,
            "matched_count": 0,
            "execution_time_seconds": execution_time_seconds,
            "latency_per_plate_ms": 0.0,
            "per_detection": []
        }

    confidences = [float(d.get("confidence", 0.0)) for d in detections_list if d.get("confidence") is not None]
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

    if not ground_truth_list:
        return {
            "average_cer": None,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "gt_recall": 0.0,
            "precision": 0.0,
            "total_edit_distance": 0,
            "average_confidence": round(avg_conf, 4),
            "correct_confidence": 0.0,
            "incorrect_confidence": 0.0,
            "total_detections": len(detections_list),
            "matched_count": 0,
            "execution_time_seconds": round(execution_time_seconds, 2),
            "latency_per_plate_ms": round((execution_time_seconds * 1000 / len(detections_list)), 1) if detections_list else 0.0,
            "per_detection": []
        }

    per_detection = []
    total_cer = 0.0
    matched_count = 0
    exact_match_count = 0
    total_edit_distance = 0
    correct_confs = []
    incorrect_confs = []
    matched_gt_set = set()

    for detection in detections_list:
        plate_text = str(detection.get("plate_number", ""))
        conf = float(detection.get("confidence", 0.0))
        if not plate_text:
            continue

        best_gt, best_cer = find_best_ground_truth_match(plate_text, ground_truth_list)

        if best_gt is not None:
            matched_count += 1
            total_cer += best_cer
            edit_dist = compute_edit_distance(plate_text, best_gt)
            total_edit_distance += edit_dist

            is_exact = (best_cer == 0.0)
            if is_exact:
                exact_match_count += 1
                correct_confs.append(conf)
                matched_gt_set.add(_normalize_plate(best_gt))
            else:
                incorrect_confs.append(conf)
                if best_cer <= 0.3:
                    matched_gt_set.add(_normalize_plate(best_gt))

            per_detection.append({
                "track_id": detection.get("track_id"),
                "plate_number": plate_text,
                "matched_ground_truth": best_gt,
                "cer": round(best_cer, 4),
                "edit_distance": edit_dist,
                "is_exact": is_exact,
                "confidence": round(conf, 4)
            })

    average_cer = (total_cer / matched_count) if matched_count > 0 else None
    exact_match_rate = (exact_match_count / matched_count) if matched_count > 0 else 0.0
    
    gt_norm_set = set(_normalize_plate(g) for g in ground_truth_list if g)
    gt_recall = (len(matched_gt_set) / len(gt_norm_set)) if gt_norm_set else 0.0
    
    precision = (len([p for p in per_detection if p["cer"] <= 0.3]) / len(per_detection)) if per_detection else 0.0

    avg_correct_conf = (sum(correct_confs) / len(correct_confs)) if correct_confs else 0.0
    avg_incorrect_conf = (sum(incorrect_confs) / len(incorrect_confs)) if incorrect_confs else 0.0
    latency_per_plate_ms = (execution_time_seconds * 1000 / len(detections_list)) if detections_list else 0.0

    return {
        "average_cer": round(average_cer, 4) if average_cer is not None else None,
        "exact_match_count": exact_match_count,
        "exact_match_rate": round(exact_match_rate, 4),
        "gt_recall": round(gt_recall, 4),
        "precision": round(precision, 4),
        "total_edit_distance": total_edit_distance,
        "average_confidence": round(avg_conf, 4),
        "correct_confidence": round(avg_correct_conf, 4),
        "incorrect_confidence": round(avg_incorrect_conf, 4),
        "total_detections": len(detections_list),
        "matched_count": matched_count,
        "execution_time_seconds": round(execution_time_seconds, 2),
        "latency_per_plate_ms": round(latency_per_plate_ms, 1),
        "per_detection": per_detection
    }


def compute_average_cer(detections_list, ground_truth_list):
    """
    Backwards-compatible wrapper that invokes compute_comprehensive_metrics.
    """
    return compute_comprehensive_metrics(detections_list, ground_truth_list)


def compute_dual_model_comparison(easyocr_detections, pytesseract_detections, ground_truth_list, easyocr_time=0.0, pytesseract_time=0.0):
    """
    Calculates side-by-side comparative Ground Truth statistics for EasyOCR vs PyTesseract.
    Determines overall model winner based on CER, Exact Match Rate, and Precision.
    """
    easy_metrics = compute_comprehensive_metrics(easyocr_detections, ground_truth_list, easyocr_time)
    tess_metrics = compute_comprehensive_metrics(pytesseract_detections, ground_truth_list, pytesseract_time)

    easy_cer = easy_metrics["average_cer"] if easy_metrics["average_cer"] is not None else 1.0
    tess_cer = tess_metrics["average_cer"] if tess_metrics["average_cer"] is not None else 1.0

    easy_exact = easy_metrics["exact_match_rate"]
    tess_exact = tess_metrics["exact_match_rate"]

    easy_prec = easy_metrics["precision"]
    tess_prec = tess_metrics["precision"]

    easy_score = (1.0 - easy_cer) * 0.4 + easy_exact * 0.4 + easy_prec * 0.2
    tess_score = (1.0 - tess_cer) * 0.4 + tess_exact * 0.4 + tess_prec * 0.2

    if abs(easy_score - tess_score) < 0.02:
        winner = "Tie"
    elif easy_score > tess_score:
        winner = "EasyOCR"
    else:
        winner = "PyTesseract"

    return {
        "winner": winner,
        "ground_truth_count": len(ground_truth_list) if ground_truth_list else 0,
        "easyocr": easy_metrics,
        "pytesseract": tess_metrics
    }
