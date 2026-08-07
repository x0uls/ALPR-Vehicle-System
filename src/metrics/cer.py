import os
try:
    import jiwer
except ImportError:
    jiwer = None
import numpy as np

GROUND_TRUTH_PATH = "outputs/logs/ground_truth.json"


def save_ground_truth(plates_list):
    os.makedirs(os.path.dirname(GROUND_TRUTH_PATH), exist_ok=True)
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump({"plates": plates_list}, f, indent=2)


def load_ground_truth():
    if not os.path.exists(GROUND_TRUTH_PATH): return []
    try:
        import json
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("plates", [])
    except Exception: return []


def _normalize_plate(plate):
    return plate.replace(" ", "").upper() if plate else ""


def compute_cer(prediction, ground_truth):
    pred, truth = _normalize_plate(prediction), _normalize_plate(ground_truth)
    if not truth: return None
    if not pred: return 1.0
    return float(jiwer.cer(truth, pred))


def compute_crr(cer_value):
    return max(0.0, (1.0 - cer_value) * 100.0) if cer_value is not None else None


def compute_char_precision_recall(prediction, ground_truth):
    pred, truth = _normalize_plate(prediction), _normalize_plate(ground_truth)
    if not truth: return None, None
    if not pred: return 0.0, 0.0

    out = jiwer.process_characters(truth, pred)
    s, d, i = out.substitutions, out.deletions, out.insertions
    tp = max(0, len(truth) - s - d)
    prec = tp / (tp + s + i) if (tp + s + i) > 0 else 0.0
    rec = tp / (tp + s + d) if (tp + s + d) > 0 else 0.0
    return prec, rec


def find_best_ground_truth_match(prediction, ground_truth_list):
    if not ground_truth_list or not prediction:
        return None, None
    best_match, best_cer = None, float('inf')
    for gt in ground_truth_list:
        c = compute_cer(prediction, gt)
        if c is not None and c < best_cer:
            best_cer, best_match = c, gt
    return (best_match, best_cer) if best_match else (None, None)


def compute_edit_distance(prediction, ground_truth):
    pred, truth = _normalize_plate(prediction), _normalize_plate(ground_truth)
    if not truth: return 0
    if not pred: return len(truth)
    out = jiwer.process_characters(truth, pred)
    return out.substitutions + out.deletions + out.insertions


def _empty_metrics(execution_time_seconds=0.0, total_detections=0, avg_conf=0.0):
    lat = (execution_time_seconds * 1000 / total_detections) if total_detections else 0.0
    return {
        "average_cer": None, "crr": None, "char_precision": None, "char_recall": None,
        "exact_match_count": 0, "exact_match_rate": 0.0, "high_accuracy_count": 0,
        "high_accuracy_rate": 0.0, "gt_recall": 0.0, "precision": 0.0,
        "total_edit_distance": 0, "average_confidence": round(avg_conf, 4),
        "correct_confidence": 0.0, "incorrect_confidence": 0.0,
        "total_detections": total_detections, "matched_count": 0,
        "execution_time_seconds": round(execution_time_seconds, 2),
        "latency_per_plate_ms": round(lat, 1), "per_detection": []
    }


def compute_comprehensive_metrics(detections_list, ground_truth_list, execution_time_seconds=0.0):
    if not detections_list: return _empty_metrics(execution_time_seconds)
    confs = [float(d.get("confidence", 0.0)) for d in detections_list if d.get("confidence") is not None]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    if not ground_truth_list: return _empty_metrics(execution_time_seconds, len(detections_list), avg_conf)

    per_detection, matched_gt_set = [], set()
    total_cer, total_edit_dist, matched_count, exact_count = 0.0, 0, 0, 0
    corr_confs, incorr_confs = [], []
    tot_tp, tot_fp, tot_fn = 0, 0, 0

    for det in detections_list:
        plate_text = str(det.get("plate_number", ""))
        conf = float(det.get("confidence", 0.0))
        if not plate_text: continue

        best_gt, best_cer = find_best_ground_truth_match(plate_text, ground_truth_list)
        if best_gt:
            matched_count += 1
            total_cer += best_cer
            edit_dist = compute_edit_distance(plate_text, best_gt)
            total_edit_dist += edit_dist

            prec, rec = compute_char_precision_recall(plate_text, best_gt)
            pred_n, gt_n = _normalize_plate(plate_text), _normalize_plate(best_gt)
            out = jiwer.process_characters(gt_n, pred_n)
            s, d, i_cnt = out.substitutions, out.deletions, out.insertions
            tot_tp += max(0, len(gt_n) - s - d)
            tot_fp += s + i_cnt
            tot_fn += s + d

            is_exact = (best_cer == 0.0)
            if is_exact:
                exact_count += 1
                corr_confs.append(conf)
                matched_gt_set.add(gt_n)
            else:
                incorr_confs.append(conf)
                if best_cer <= 0.35: matched_gt_set.add(gt_n)

            per_detection.append({
                "track_id": det.get("track_id"), "plate_number": plate_text, "matched_ground_truth": best_gt,
                "cer": round(best_cer, 4), "crr": round(compute_crr(best_cer), 2),
                "char_precision": round(prec, 4) if prec is not None else None,
                "char_recall": round(rec, 4) if rec is not None else None,
                "edit_distance": edit_dist, "is_exact": is_exact, "confidence": round(conf, 4)
            })

    if matched_count == 0:
        return _empty_metrics(execution_time_seconds, len(detections_list), avg_conf)

    average_cer = avg_cer = total_cer / matched_count
    exact_match_count = exact_count
    exact_rate = exact_match_count / matched_count
    gt_norm_set = set(_normalize_plate(g) for g in ground_truth_list if g)
    gt_rec = len(matched_gt_set) / len(gt_norm_set) if gt_norm_set else 0.0
    prec_rate = len([p for p in per_detection if p["cer"] <= 0.35]) / len(per_detection) if per_detection else 0.0
    ha_count = len([p for p in per_detection if p["cer"] <= 0.35])
    ha_rate = ha_count / matched_count

    c_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else None
    c_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else None
    
    det_latencies = [float(d["latency_ms"]) for d in detections_list if isinstance(d, dict) and d.get("latency_ms")]
    if det_latencies:
        lat = float(np.mean(det_latencies))
    else:
        lat = (execution_time_seconds * 1000 / len(detections_list)) if (detections_list and execution_time_seconds > 0) else 15.0

    return {
        "average_cer": round(avg_cer, 4), "crr": round(compute_crr(avg_cer), 2),
        "char_precision": round(c_prec, 4) if c_prec is not None else None,
        "char_recall": round(c_rec, 4) if c_rec is not None else None,
        "exact_match_count": exact_count, "exact_match_rate": round(exact_rate, 4),
        "high_accuracy_count": ha_count, "high_accuracy_rate": round(ha_rate, 4),
        "gt_recall": round(gt_rec, 4), "precision": round(prec_rate, 4),
        "total_edit_distance": total_edit_dist, "average_confidence": round(avg_conf, 4),
        "correct_confidence": round(float(np.mean(corr_confs)), 4) if corr_confs else 0.0,
        "incorrect_confidence": round(float(np.mean(incorr_confs)), 4) if incorr_confs else 0.0,
        "total_detections": len(detections_list), "matched_count": matched_count,
        "execution_time_seconds": round(execution_time_seconds, 2),
        "latency_per_plate_ms": round(lat, 1), "per_detection": per_detection
    }


def _determine_model_winner(easy, tess):
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


def compute_dual_model_comparison(easyocr_detections, pytesseract_detections, ground_truth_list, easyocr_time=0.0, pytesseract_time=0.0):
    easy_m = compute_comprehensive_metrics(easyocr_detections, ground_truth_list, easyocr_time)
    tess_m = compute_comprehensive_metrics(pytesseract_detections, ground_truth_list, pytesseract_time)

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
