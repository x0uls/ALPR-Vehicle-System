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
import numpy as np

GROUND_TRUTH_PATH = "outputs/logs/ground_truth.json"


def _normalize_plate(plate):
    """Normalizes plate text by stripping spaces and converting to uppercase."""
    return plate.replace(" ", "").upper() if plate else ""


def compute_cer(prediction, ground_truth):
    """
    Computes Character Error Rate (CER) between predicted plate and ground truth.
    Returns float (0.0 = perfect match).
    """
    pred_normalized = _normalize_plate(prediction)
    truth_normalized = _normalize_plate(ground_truth)
    
    if not truth_normalized:
        return None
    if not pred_normalized:
        return 1.0
    
    # jiwer.cer(truth, pred): Levenshtein character error rate
    return float(jiwer.cer(truth_normalized, pred_normalized))


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


def _empty_metrics(execution_time_seconds=0.0, total_detections=0, avg_conf=0.0):
    """Returns a zeroed-out metrics dictionary for edge cases (no detections or no ground truth)."""
    latency = (execution_time_seconds * 1000 / total_detections) if total_detections else 0.0
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
        "total_detections": total_detections,
        "matched_count": 0,
        "execution_time_seconds": round(execution_time_seconds, 2),
        "latency_per_plate_ms": round(latency, 1),
        "per_detection": []
    }


def _match_detections(detections_list, ground_truth_list):
    """Matches each detection to its closest ground truth and returns per-detection results."""
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

    return {
        "per_detection": per_detection,
        "total_cer": total_cer,
        "matched_count": matched_count,
        "exact_match_count": exact_match_count,
        "total_edit_distance": total_edit_distance,
        "correct_confs": correct_confs,
        "incorrect_confs": incorrect_confs,
        "matched_gt_set": matched_gt_set,
    }


def compute_comprehensive_metrics(detections_list, ground_truth_list, execution_time_seconds=0.0):
    """
    Computes comprehensive Ground Truth evaluation metrics for an OCR model's detections.
    Orchestrates: early return → match → aggregate.
    """
    if not detections_list:
        return _empty_metrics(execution_time_seconds)

    confidences = [float(d.get("confidence", 0.0)) for d in detections_list if d.get("confidence") is not None]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    if not ground_truth_list:
        return _empty_metrics(execution_time_seconds, len(detections_list), avg_conf)

    results = _match_detections(detections_list, ground_truth_list)

    matched_count = results["matched_count"]
    exact_match_count = results["exact_match_count"]
    per_detection = results["per_detection"]
    matched_gt_set = results["matched_gt_set"]

    average_cer = (results["total_cer"] / matched_count) if matched_count > 0 else None
    exact_match_rate = (exact_match_count / matched_count) if matched_count > 0 else 0.0
    
    gt_norm_set = set(_normalize_plate(g) for g in ground_truth_list if g)
    gt_recall = (len(matched_gt_set) / len(gt_norm_set)) if gt_norm_set else 0.0
    
    precision = (len([p for p in per_detection if p["cer"] <= 0.3]) / len(per_detection)) if per_detection else 0.0

    avg_correct_conf = float(np.mean(results["correct_confs"])) if results["correct_confs"] else 0.0
    avg_incorrect_conf = float(np.mean(results["incorrect_confs"])) if results["incorrect_confs"] else 0.0
    latency_per_plate_ms = (execution_time_seconds * 1000 / len(detections_list)) if detections_list else 0.0

    return {
        "average_cer": round(average_cer, 4) if average_cer is not None else None,
        "exact_match_count": exact_match_count,
        "exact_match_rate": round(exact_match_rate, 4),
        "gt_recall": round(gt_recall, 4),
        "precision": round(precision, 4),
        "total_edit_distance": results["total_edit_distance"],
        "average_confidence": round(avg_conf, 4),
        "correct_confidence": round(avg_correct_conf, 4),
        "incorrect_confidence": round(avg_incorrect_conf, 4),
        "total_detections": len(detections_list),
        "matched_count": matched_count,
        "execution_time_seconds": round(execution_time_seconds, 2),
        "latency_per_plate_ms": round(latency_per_plate_ms, 1),
        "per_detection": per_detection
    }


def _determine_model_winner(easy_metrics, tess_metrics):
    """Computes weighted composite scores and returns the winning model name."""
    easy_count = easy_metrics.get("total_detections", 0)
    tess_count = tess_metrics.get("total_detections", 0)

    if easy_count == 0 and tess_count > 0:
        return "PyTesseract"
    if tess_count == 0 and easy_count > 0:
        return "EasyOCR"
    if easy_count == 0 and tess_count == 0:
        return "Tie"

    easy_cer = easy_metrics["average_cer"] if easy_metrics["average_cer"] is not None else 1.0
    tess_cer = tess_metrics["average_cer"] if tess_metrics["average_cer"] is not None else 1.0

    easy_score = (1.0 - easy_cer) * 0.4 + easy_metrics["exact_match_rate"] * 0.4 + easy_metrics["precision"] * 0.2
    tess_score = (1.0 - tess_cer) * 0.4 + tess_metrics["exact_match_rate"] * 0.4 + tess_metrics["precision"] * 0.2

    if abs(easy_score - tess_score) < 0.02:
        return "Tie"
    return "EasyOCR" if easy_score > tess_score else "PyTesseract"


def compute_dual_model_comparison(easyocr_detections, pytesseract_detections, ground_truth_list, easyocr_time=0.0, pytesseract_time=0.0):
    """
    Calculates side-by-side comparative Ground Truth statistics for EasyOCR vs PyTesseract.
    Determines overall model winner based on CER, Exact Match Rate, and Precision.
    """
    easy_metrics = compute_comprehensive_metrics(easyocr_detections, ground_truth_list, easyocr_time)
    tess_metrics = compute_comprehensive_metrics(pytesseract_detections, ground_truth_list, pytesseract_time)

    return {
        "winner": _determine_model_winner(easy_metrics, tess_metrics),
        "ground_truth_count": len(ground_truth_list) if ground_truth_list else 0,
        "easyocr": easy_metrics,
        "pytesseract": tess_metrics
    }
