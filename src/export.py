import glob
import os
from io import BytesIO
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
from src.metrics.cer import find_best_ground_truth_match, compute_average_cer

from src.metrics.cer import find_best_ground_truth_match, compute_comprehensive_metrics, compute_dual_model_comparison


def _populate_detection_sheet(worksheet, df: pd.DataFrame, ground_truth_plates, title_prefix=""):
    headers = ["Track ID", "Timestamp", "Vehicle Type", "Color", "Plate Number", "Confidence", "Vehicle Image", "Plate Crop"]
    if ground_truth_plates:
        headers.extend(["Matched GT", "CER"])
    worksheet.append(headers)

    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center")

    for column_index in range(1, len(headers) + 1):
        cell = worksheet.cell(row=1, column=column_index)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align

    worksheet.row_dimensions[1].height = 25
    font_regular = Font(name="Arial", size=10)
    align_center = Alignment(horizontal="center", vertical="center")

    if df.empty:
        return

    for idx, row in df.iterrows():
        row_index = idx + 2
        track_id = row.get("track_id", "")
        timestamp = row.get("timestamp", "")
        vehicle_type = row.get("vehicle_type", "")
        color = row.get("color", "")
        plate_number = row.get("plate_number", "")
        confidence = row.get("confidence", 0.0)
        snapshot_path = row.get("snapshot_path", "")

        worksheet.cell(row=row_index, column=1, value=int(track_id) if str(track_id).isdigit() else track_id)
        worksheet.cell(row=row_index, column=2, value=str(timestamp))
        worksheet.cell(row=row_index, column=3, value=str(vehicle_type).capitalize())
        worksheet.cell(row=row_index, column=4, value=str(color).capitalize() if pd.notna(color) else "")
        worksheet.cell(row=row_index, column=5, value=str(plate_number))

        confidence_value = float(confidence) if pd.notna(confidence) else 0.0
        confidence_cell = worksheet.cell(row=row_index, column=6, value=confidence_value)
        confidence_cell.number_format = "0%"

        for column_index in range(1, 7):
            cell = worksheet.cell(row=row_index, column=column_index)
            cell.font = font_regular
            cell.alignment = align_center

        if ground_truth_plates and plate_number:
            best_gt, best_cer = find_best_ground_truth_match(str(plate_number), ground_truth_plates)
            gt_cell = worksheet.cell(row=row_index, column=9, value=str(best_gt) if best_gt else "No Match")
            gt_cell.font = font_regular
            gt_cell.alignment = align_center

            if best_cer is not None:
                cer_cell = worksheet.cell(row=row_index, column=10, value=best_cer)
                cer_cell.number_format = "0.00%"
                cer_cell.font = font_regular
                cer_cell.alignment = align_center

                if best_cer == 0.0:
                    cer_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif best_cer <= 0.3:
                    cer_cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                else:
                    cer_cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
            else:
                worksheet.cell(row=row_index, column=10, value="N/A").font = font_regular
        elif ground_truth_plates:
            worksheet.cell(row=row_index, column=9, value="").font = font_regular
            worksheet.cell(row=row_index, column=10, value="").font = font_regular

        worksheet.row_dimensions[row_index].height = 80

        crop_glob = f"outputs/plate_crops/Processed/*_track{track_id}_processed.jpg"
        matching_files = glob.glob(crop_glob)
        if matching_files and os.path.exists(matching_files[0]):
            try:
                img_plate = OpenpyxlImage(matching_files[0])
                img_plate.width = 100
                img_plate.height = 35
                worksheet.add_image(img_plate, f"H{row_index}")
            except Exception:
                worksheet.cell(row=row_index, column=8, value="Img Err")
        else:
            worksheet.cell(row=row_index, column=8, value="No Crop")

        if snapshot_path and os.path.exists(snapshot_path):
            try:
                img_vehicle = OpenpyxlImage(snapshot_path)
                img_vehicle.width = 120
                img_vehicle.height = 70
                worksheet.add_image(img_vehicle, f"G{row_index}")
            except Exception:
                worksheet.cell(row=row_index, column=7, value="Img Err")
        else:
            worksheet.cell(row=row_index, column=7, value="No Image")

    for column in worksheet.columns:
        max_len = 0
        column_letter = get_column_letter(column[0].column)
        if column_letter == "G":
            worksheet.column_dimensions["G"].width = 20
            continue
        if column_letter == "H":
            worksheet.column_dimensions["H"].width = 18
            continue
        for cell in column:
            value_string = str(cell.value or '')
            if len(value_string) > max_len:
                max_len = len(value_string)
        worksheet.column_dimensions[column_letter].width = max(max_len + 3, 10)


def build_xlsx_report(easy_df: pd.DataFrame, tess_df: pd.DataFrame = None, ground_truth_plates=None) -> BytesIO:
    """
    Generates a multi-tab Excel report comparing EasyOCR and PyTesseract model performance against Ground Truth.
    """
    if tess_df is None and isinstance(easy_df, pd.DataFrame):
        tess_df = pd.DataFrame()

    if ground_truth_plates is None:
        ground_truth_plates = []

    workbook = openpyxl.Workbook()
    
    # ── Sheet 1: Executive Comparison Summary ──
    comp_ws = workbook.active
    comp_ws.title = "Model Comparison"

    comp_ws.cell(row=1, column=1, value="ALPR Dual-Model Ground Truth Evaluation").font = Font(name="Arial", size=14, bold=True, color="1F497D")
    comp_ws.merge_cells("A1:D1")

    easy_dets = easy_df.to_dict(orient="records") if not easy_df.empty else []
    tess_dets = tess_df.to_dict(orient="records") if not tess_df.empty else []

    easy_metrics = compute_comprehensive_metrics(easy_dets, ground_truth_plates)
    tess_metrics = compute_comprehensive_metrics(tess_dets, ground_truth_plates)

    comp_headers = ["Metric / Benchmark", "EasyOCR", "PyTesseract", "Winner / Difference"]
    for col_idx, hdr in enumerate(comp_headers, 1):
        cell = comp_ws.cell(row=3, column=col_idx, value=hdr)
        cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center")

    comp_ws.row_dimensions[3].height = 24

    metrics_rows = [
        ("Average Character Error Rate (CER)", easy_metrics["average_cer"], tess_metrics["average_cer"], "percentage_lower"),
        ("Exact Match Rate (CER = 0.0)", easy_metrics["exact_match_rate"], tess_metrics["exact_match_rate"], "percentage_higher"),
        ("Ground Truth Recall (Coverage)", easy_metrics["gt_recall"], tess_metrics["gt_recall"], "percentage_higher"),
        ("Precision (Valid Matches)", easy_metrics["precision"], tess_metrics["precision"], "percentage_higher"),
        ("Total Character Edit Distance", easy_metrics["total_edit_distance"], tess_metrics["total_edit_distance"], "int_lower"),
        ("Average Confidence Rating", easy_metrics["average_confidence"], tess_metrics["average_confidence"], "percentage_higher"),
        ("Confidence on Correct Reads", easy_metrics["correct_confidence"], tess_metrics["correct_confidence"], "percentage_higher"),
        ("Total Vehicles Logged", easy_metrics["total_detections"], tess_metrics["total_detections"], "int_neutral"),
    ]

    font_label = Font(name="Arial", size=10, bold=True)
    font_val = Font(name="Arial", size=10)
    align_center = Alignment(horizontal="center", vertical="center")

    for idx, (label, e_val, t_val, metric_type) in enumerate(metrics_rows, 4):
        comp_ws.cell(row=idx, column=1, value=label).font = font_label

        c_e = comp_ws.cell(row=idx, column=2, value=e_val if e_val is not None else "N/A")
        c_t = comp_ws.cell(row=idx, column=3, value=t_val if t_val is not None else "N/A")
        c_e.font = font_val
        c_t.font = font_val
        c_e.alignment = align_center
        c_t.alignment = align_center

        if "percentage" in metric_type and isinstance(e_val, (int, float)):
            c_e.number_format = "0.00%"
        if "percentage" in metric_type and isinstance(t_val, (int, float)):
            c_t.number_format = "0.00%"

        # Determine row winner
        if e_val is not None and t_val is not None:
            if metric_type == "percentage_lower" or metric_type == "int_lower":
                winner_str = "EasyOCR" if e_val < t_val else ("PyTesseract" if t_val < e_val else "Tie")
            elif metric_type == "percentage_higher":
                winner_str = "EasyOCR" if e_val > t_val else ("PyTesseract" if t_val > e_val else "Tie")
            else:
                winner_str = "--"
        else:
            winner_str = "--"

        c_w = comp_ws.cell(row=idx, column=4, value=winner_str)
        c_w.font = Font(name="Arial", size=10, bold=True, color="1E3A8A" if winner_str=="EasyOCR" else "6B21A8" if winner_str=="PyTesseract" else "475569")
        c_w.alignment = align_center

    for col in range(1, 5):
        comp_ws.column_dimensions[get_column_letter(col)].width = 28

    # ── Sheet 2: EasyOCR Detections ──
    easy_ws = workbook.create_sheet(title="EasyOCR Detections")
    _populate_detection_sheet(easy_ws, easy_df, ground_truth_plates, "EasyOCR")

    # ── Sheet 3: PyTesseract Detections ──
    tess_ws = workbook.create_sheet(title="PyTesseract Detections")
    _populate_detection_sheet(tess_ws, tess_df, ground_truth_plates, "PyTesseract")

    out_buf = BytesIO()
    workbook.save(out_buf)
    out_buf.seek(0)
    return out_buf

