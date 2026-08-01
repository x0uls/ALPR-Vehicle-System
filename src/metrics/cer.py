"""
Evaluation metrics for OCR accuracy benchmarking.

Provides six metrics for evaluating license plate OCR performance:

1. Exact Match Accuracy (string-level): Correctly Predicted / Total × 100%
2. Character Error Rate (CER): (Substitutions + Deletions + Insertions) / N
3. Character Recognition Rate (CRR): (1 - CER) × 100%
4. Character Precision: TP / (TP + FP) — penalizes hallucinated characters
5. Character Recall: TP / (TP + FN) — penalizes missed characters
6. Inference Latency: Average execution time per image (ms)

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

4. Character Precision and Recall use the same S/D/I counts:
   - TP (True Positives) = N - S - D  (ground truth chars matched correctly)
   - FP (False Positives) = S + I      (wrong chars + extra hallucinated chars)
   - FN (False Negatives) = S + D      (wrong chars + missed chars)
   - Precision = TP / (TP + FP)
   - Recall    = TP / (TP + FN)
"""

import os
import csv
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


def compute_crr(cer_value):
    """
    Computes Character Recognition Rate (CRR) from a CER value.
    CRR = (1 - CER) × 100%, clamped to a minimum of 0.
    """
    if cer_value is None:
        return None
    return max(0.0, (1.0 - cer_value) * 100.0)


def compute_char_precision_recall(prediction, ground_truth):
    """
    Computes character-level Precision and Recall between predicted and ground truth plates.

    Uses jiwer.process_characters() to obtain Substitution, Deletion, and Insertion counts,
    then derives True Positives (TP), False Positives (FP), and False Negatives (FN):

        TP = N - S - D     (ground truth chars matched correctly)
        FP = S + I         (substituted + inserted chars — hallucinated output)
        FN = S + D         (substituted + deleted chars — missed input)

    Returns (precision, recall) as floats in [0.0, 1.0], or (None, None) if inputs are empty.
    """
    pred_norm = _normalize_plate(prediction)
    truth_norm = _normalize_plate(ground_truth)

    if not truth_norm:
        return None, None
    if not pred_norm:
        return 0.0, 0.0

    output = jiwer.process_characters(truth_norm, pred_norm)
    s, d, i = output.substitutions, output.deletions, output.insertions
    n = len(truth_norm)

    tp = max(0, n - s - d)
    fp = s + i
    fn = s + d

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


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


def load_ground_truth_csv(csv_path):
    """
    Loads ground truth from a CSV file with columns: filename, ground_truth.
    Returns a dict mapping filename → ground_truth_plate (normalized uppercase, stripped).
    """
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = (row.get("filename") or "").strip()
                gt = (row.get("ground_truth") or "").strip().upper()
                if fname and gt:
                    mapping[fname] = gt
    except (IOError, csv.Error):
        pass
    return mapping


def save_ground_truth_csv(mapping, csv_path):
    """
    Writes a filename → ground_truth mapping to a CSV file.
    """
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ground_truth"])
        for fname, gt in mapping.items():
            writer.writerow([fname, gt])


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
        "crr": None,
        "char_precision": None,
        "char_recall": None,
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
    # Character-level accumulators for Precision and Recall
    total_tp = 0
    total_fp = 0
    total_fn = 0

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

            # Accumulate character-level TP/FP/FN
            pred_norm = _normalize_plate(plate_text)
            gt_norm = _normalize_plate(best_gt)
            output = jiwer.process_characters(gt_norm, pred_norm)
            s, d, i_count = output.substitutions, output.deletions, output.insertions
            n = len(gt_norm)
            tp = max(0, n - s - d)
            total_tp += tp
            total_fp += s + i_count
            total_fn += s + d

            char_prec, char_rec = compute_char_precision_recall(plate_text, best_gt)

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
                "crr": round(compute_crr(best_cer), 2),
                "char_precision": round(char_prec, 4) if char_prec is not None else None,
                "char_recall": round(char_rec, 4) if char_rec is not None else None,
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
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
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

    # Aggregate character-level Precision and Recall from accumulated TP/FP/FN
    agg_tp, agg_fp, agg_fn = results["total_tp"], results["total_fp"], results["total_fn"]
    char_precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else None
    char_recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else None

    avg_correct_conf = float(np.mean(results["correct_confs"])) if results["correct_confs"] else 0.0
    avg_incorrect_conf = float(np.mean(results["incorrect_confs"])) if results["incorrect_confs"] else 0.0
    latency_per_plate_ms = (execution_time_seconds * 1000 / len(detections_list)) if detections_list else 0.0

    return {
        "average_cer": round(average_cer, 4) if average_cer is not None else None,
        "crr": round(compute_crr(average_cer), 2) if average_cer is not None else None,
        "char_precision": round(char_precision, 4) if char_precision is not None else None,
        "char_recall": round(char_recall, 4) if char_recall is not None else None,
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

    chart_url = None
    try:
        from src.metrics.visualization import generate_benchmark_charts
        chart_url = generate_benchmark_charts(easy_metrics, tess_metrics)
    except Exception as e:
        print(f"[MATPLOTLIB CHART ERROR] {e}")

    return {
        "winner": _determine_model_winner(easy_metrics, tess_metrics),
        "ground_truth_count": len(ground_truth_list) if ground_truth_list else 0,
        "easyocr": easy_metrics,
        "pytesseract": tess_metrics,
        "chart_url": chart_url
    }
