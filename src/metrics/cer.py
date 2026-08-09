import os
import json
import numpy as np

# jiwer is a library that computes text similarity metrics (originally built for
# speech-to-text evaluation, but works well for OCR too). It handles the hard part —
# calculating edit distance (substitutions, deletions, insertions) between two strings
# using the Levenshtein algorithm — so we don't have to implement that ourselves.
try:
    import jiwer
except ImportError:
    jiwer = None

GROUND_TRUTH_PATH = "outputs/logs/ground_truth.json"


def save_ground_truth(plates_list):
    """Saves the list of correct/expected plate numbers to a JSON file on disk."""
    os.makedirs(os.path.dirname(GROUND_TRUTH_PATH), exist_ok=True)
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump({"plates": plates_list}, f, indent=2)


def load_ground_truth():
    """Loads the saved ground truth plate list back from disk. Returns empty list if missing/corrupt."""
    if not os.path.exists(GROUND_TRUTH_PATH):
        return []
    try:
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("plates", [])
    except Exception:
        return []


def _normalize(plate):
    """
    Cleans up a plate string before comparison: removes spaces, converts to uppercase.
    Why: "WD 586 D" and "wd586d" should be treated as the SAME plate when comparing,
    since spacing/casing differences aren't real OCR errors.
    """
    return plate.replace(" ", "").upper() if plate else ""


def compute_cer(prediction, ground_truth):
    """
    CER = Character Error Rate.
    Answers: "Out of all the characters in the correct answer, how many edits
    (wrong/missing/extra characters) did it take to turn the OCR's guess into the truth?"

    Formula (handled internally by jiwer): (substitutions + deletions + insertions) / length of ground truth
    Lower is better. 0.0 = perfect read. 1.0 = completely wrong (as many errors as there are characters).
    """
    pred, truth = _normalize(prediction), _normalize(ground_truth)
    if not truth:
        return None  # can't score against an empty/missing ground truth
    if not pred:
        return 1.0  # OCR produced nothing = 100% error rate
    return float(jiwer.cer(truth, pred)) if jiwer else 1.0


def compute_crr(cer_val):
    """
    CRR = Character Recognition Rate. Just CER flipped into a "% correct" framing
    instead of "% wrong", expressed as a percentage (0-100) instead of a fraction (0-1).
    CRR = (1 - CER) * 100. Higher is better.
    """
    return max(0.0, (1.0 - cer_val) * 100.0) if cer_val is not None else None


def _empty_metrics(exec_time=0.0, total_dets=0, avg_conf=0.0):
    """
    Returns a metrics dict filled with zero/null values.
    Used as a fallback when there's nothing to score (no detections, or no ground truth to compare against),
    so the rest of the dashboard code always has a consistent dict shape to work with (no missing keys/crashes).
    """
    lat = (exec_time * 1000 / total_dets) if total_dets else 0.0
    return {
        "average_cer": None, "crr": None, "char_precision": None, "char_recall": None,
        "exact_match_count": 0, "exact_match_rate": 0.0, "high_accuracy_count": 0,
        "high_accuracy_rate": 0.0, "gt_recall": 0.0, "precision": 0.0,
        "total_edit_distance": 0, "average_confidence": round(avg_conf, 4),
        "correct_confidence": 0.0, "incorrect_confidence": 0.0,
        "total_detections": total_dets, "matched_count": 0,
        "execution_time_seconds": round(exec_time, 2),
        "latency_per_plate_ms": round(lat, 1), "per_detection": []
    }


def compute_comprehensive_metrics(detections, ground_truth_list, exec_time=0.0):
    """
    Main scoring function. Runs once per OCR engine (once for EasyOCR's results,
    once for PyTesseract's results). Compares every detected plate against its
    matching ground truth entry and aggregates all the accuracy metrics.
    """
    if not detections:
        return _empty_metrics(exec_time)

    # Average OCR confidence score across all detections (regardless of whether they were correct)
    confs = [float(d.get("confidence", 0.0)) for d in detections if d.get("confidence") is not None]
    avg_conf = float(np.mean(confs)) if confs else 0.0

    if not ground_truth_list:
        return _empty_metrics(exec_time, len(detections), avg_conf)

    per_det, matched_gt_set = [], set()
    total_cer, total_edit_dist, matched_cnt, exact_cnt = 0.0, 0, 0, 0
    corr_confs, incorr_confs = [], []
    # tot_tp/fp/fn = running totals across ALL detections, used later for overall precision/recall
    # tp = true positive (correct char), fp = false positive (wrong/extra char), fn = false negative (missed char)
    tot_tp, tot_fp, tot_fn = 0, 0, 0

    for i, det in enumerate(detections):
        plate_text = str(det.get("plate_number", ""))
        conf = float(det.get("confidence", 0.0))
        if not plate_text:
            continue  # skip empty/failed OCR reads

        # Figure out which ground truth entry this detection should be compared against.
        # Tries several possible field names / lookup strategies depending on how the
        # detection data was structured (dict keyed by filename, list in matching order, etc.)
        target_gt = det.get("matched_ground_truth") or det.get("ground_truth") or det.get("gt")
        if not target_gt and isinstance(ground_truth_list, dict):
            target_gt = ground_truth_list.get(det.get("file_name"))
        elif not target_gt and isinstance(ground_truth_list, list) and i < len(ground_truth_list):
            target_gt = ground_truth_list[i]

        if target_gt and target_gt != '--':
            best_gt = target_gt
            best_cer = compute_cer(plate_text, best_gt)
            matched_cnt += 1
            total_cer += best_cer

            pred_n, gt_n = _normalize(plate_text), _normalize(best_gt)

            # jiwer.process_characters does the actual character-by-character alignment
            # (Levenshtein) and tells us exactly what KIND of errors happened:
            #   s = substitutions (wrong character in the right spot, e.g. "6" read as "G")
            #   d = deletions     (a character that should be there but OCR missed it entirely)
            #   i_cnt = insertions (an extra character OCR added that shouldn't be there)
            out = jiwer.process_characters(gt_n, pred_n) if jiwer else None
            s = out.substitutions if out else 0
            d = out.deletions if out else 0
            i_cnt = out.insertions if out else 0
            edit_dist = s + d + i_cnt  # total number of character-level fixes needed
            total_edit_dist += edit_dist

            # tp = correctly recognized characters = (length of truth) minus the characters
            # that were wrong (substitutions) or missing (deletions)
            tp = max(0, len(gt_n) - s - d)
            tot_tp += tp
            tot_fp += (s + i_cnt)  # "false positives" = wrong chars + extra chars OCR shouldn't have output
            tot_fn += (s + d)      # "false negatives" = wrong chars + chars OCR failed to detect at all

            # PRECISION (this detection only): of everything OCR said, how much was correct?
            #   = correct / (correct + wrong + extra)
            prec = tp / (tp + s + i_cnt) if (tp + s + i_cnt) > 0 else 0.0

            # RECALL (this detection only): of everything that should be there, how much did OCR catch?
            #   = correct / (correct + wrong + missed)
            rec = tp / (tp + s + d) if (tp + s + d) > 0 else 0.0

            is_exact = (best_cer == 0.0)  # perfect read, zero errors

            if is_exact:
                exact_cnt += 1
                corr_confs.append(conf)       # track confidence scores of CORRECT reads
                matched_gt_set.add(gt_n)
            else:
                incorr_confs.append(conf)     # track confidence scores of INCORRECT reads
                # Even if not a perfect match, count it as "close enough" if error rate is low (<=35%)
                if best_cer <= 0.35:
                    matched_gt_set.add(gt_n)

            # Store a detailed record for this single detection (useful for the per-plate table in your dashboard)
            per_det.append({
                "track_id": det.get("track_id"), "plate_number": plate_text, "matched_ground_truth": best_gt,
                "cer": round(best_cer, 4), "crr": round(compute_crr(best_cer), 2),
                "char_precision": round(prec, 4), "char_recall": round(rec, 4),
                "edit_distance": edit_dist, "is_exact": is_exact, "confidence": round(conf, 4)
            })

    if matched_cnt == 0:
        return _empty_metrics(exec_time, len(detections), avg_conf)

    # ---- Aggregate stats across ALL detections for this engine ----

    avg_cer = total_cer / matched_cnt  # average error rate across every matched plate

    # gt_recall: of all the UNIQUE plates in the ground truth set, what fraction did we
    # successfully detect at least once (exact or close match)? Different from character recall —
    # this is asking "did we find this plate at all," not "how many characters were right."
    gt_norm_set = set(_normalize(g) for g in ground_truth_list if g)
    gt_rec = len(matched_gt_set) / len(gt_norm_set) if gt_norm_set else 0.0

    # Count of detections considered "high accuracy" (CER <= 0.35, i.e. mostly correct even if not perfect)
    ha_cnt = len([p for p in per_det if p["cer"] <= 0.35])

    # Overall PRECISION and RECALL across the whole dataset (not per-detection — the totals)
    c_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else None
    c_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else None

    # Latency: how long (in ms) each plate took to process, on average
    det_lats = [float(d["latency_ms"]) for d in detections if isinstance(d, dict) and d.get("latency_ms")]
    lat = float(np.mean(det_lats)) if det_lats else ((exec_time * 1000 / len(detections)) if (detections and exec_time > 0) else 15.0)

    return {
        "average_cer": round(avg_cer, 4),
        "crr": round(compute_crr(avg_cer), 2),
        "char_precision": round(c_prec, 4) if c_prec is not None else None,
        "char_recall": round(c_rec, 4) if c_rec is not None else None,
        "exact_match_count": exact_cnt,                              # how many plates were read PERFECTLY
        "exact_match_rate": round(exact_cnt / matched_cnt, 4),       # % of plates read perfectly
        "high_accuracy_count": ha_cnt,                                # how many plates were "close enough" (CER<=0.35)
        "high_accuracy_rate": round(ha_cnt / matched_cnt, 4),
        "gt_recall": round(gt_rec, 4),                                # % of unique ground truth plates found at all
        "precision": round(ha_cnt / len(per_det), 4) if per_det else 0.0,  # simplified "plate-level" precision
        "total_edit_distance": total_edit_dist,
        "average_confidence": round(avg_conf, 4),
        "correct_confidence": round(float(np.mean(corr_confs)), 4) if corr_confs else 0.0,   # avg confidence when RIGHT
        "incorrect_confidence": round(float(np.mean(incorr_confs)), 4) if incorr_confs else 0.0,  # avg confidence when WRONG
        "total_detections": len(detections),
        "matched_count": matched_cnt,
        "execution_time_seconds": round(exec_time, 2),
        "latency_per_plate_ms": round(lat, 1),
        "per_detection": per_det  # full breakdown, one entry per plate — powers your dashboard's detail table
    }


def _determine_model_winner(easy, tess):
    """
    Decides which OCR engine performed better overall using a WEIGHTED SCORE:
        score = (1 - CER) * 0.4  +  exact_match_rate * 0.4  +  precision * 0.2

    Why weighted instead of a plain average: CER-accuracy and exact-match rate are
    considered the most important signals (40% weight each), while precision is
    treated as a secondary/supporting signal (only 20% weight).
    The weights sum to 1.0 so the final score stays in a clean 0-1 range.

    If the two scores are within 0.02 of each other, it's called a "Tie" rather than
    picking a winner based on a negligible difference.
    """
    e_cnt, t_cnt = easy.get("total_detections", 0), tess.get("total_detections", 0)
    if e_cnt == 0 and t_cnt > 0: return "PyTesseract"
    if t_cnt == 0 and e_cnt > 0: return "EasyOCR"
    if e_cnt == 0 and t_cnt == 0: return "Tie"

    e_cer = easy["average_cer"] if easy["average_cer"] is not None else 1.0
    t_cer = tess["average_cer"] if tess["average_cer"] is not None else 1.0

    e_score = (1.0 - e_cer) * 0.4 + easy["exact_match_rate"] * 0.4 + easy["precision"] * 0.2
    t_score = (1.0 - t_cer) * 0.4 + tess["exact_match_rate"] * 0.4 + tess["precision"] * 0.2

    if abs(e_score - t_score) < 0.02: return "Tie"
    return "EasyOCR" if e_score > t_score else "PyTesseract"


def compute_dual_model_comparison(easyocr_dets, pytesseract_dets, ground_truth_list, easyocr_time=0.0, pytesseract_time=0.0):
    """
    Top-level entry point for the whole benchmark comparison.
    Runs the full metrics calculation once for each engine, decides a winner,
    and (if the visualization module is available) generates the comparison charts
    shown on your dashboard.
    """
    easy_m = compute_comprehensive_metrics(easyocr_dets, ground_truth_list, easyocr_time)
    tess_m = compute_comprehensive_metrics(pytesseract_dets, ground_truth_list, pytesseract_time)

    chart_url = None
    try:
        from src.metrics.visualization import generate_benchmark_charts
        chart_url = generate_benchmark_charts(easy_m, tess_m)
    except Exception as e:
        print(f"[MATPLOTLIB CHART ERROR] {e}")

    return {
        "winner": _determine_model_winner(easy_m, tess_m),
        "ground_truth_count": len(ground_truth_list) if ground_truth_list else 0,
        "easyocr": easy_m,
        "pytesseract": tess_m,
        "chart_url": chart_url
    }