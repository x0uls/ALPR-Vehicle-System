import glob
import os
from io import BytesIO
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.styles import Font, Alignment, PatternFill

def build_xlsx_report(df: pd.DataFrame) -> BytesIO:
    """Generate an Excel workbook with embedded license plate crops from detection DataFrame."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ALPR Detections"
    
    headers = ["Track ID", "Timestamp", "Vehicle Type", "Color", "Plate Number", "Confidence", "Vehicle Image", "Plate Crop"]
    ws.append(headers)
    
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center")
    
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        
    ws.row_dimensions[1].height = 25
    
    font_regular = Font(name="Arial", size=10)
    align_center = Alignment(horizontal="center", vertical="center")
    
    for idx, row in df.iterrows():
        row_idx = idx + 2
        
        track_id = row.get("track_id", "")
        timestamp = row.get("timestamp", "")
        vehicle_type = row.get("vehicle_type", "")
        color = row.get("color", "")
        plate_number = row.get("plate_number", "")
        confidence = row.get("confidence", 0.0)
        snapshot_path = row.get("snapshot_path", "")
        
        ws.cell(row=row_idx, column=1, value=int(track_id) if str(track_id).isdigit() else track_id)
        ws.cell(row=row_idx, column=2, value=str(timestamp))
        ws.cell(row=row_idx, column=3, value=str(vehicle_type).capitalize())
        ws.cell(row=row_idx, column=4, value=str(color).capitalize() if pd.notna(color) else "")
        ws.cell(row=row_idx, column=5, value=str(plate_number))
        
        conf_val = float(confidence) if pd.notna(confidence) else 0.0
        conf_cell = ws.cell(row=row_idx, column=6, value=conf_val)
        conf_cell.number_format = "0%"
        
        for col_idx in range(1, 7):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = font_regular
            cell.alignment = align_center
        
        ws.row_dimensions[row_idx].height = 80
        
        # Locate corresponding processed plate crop for track
        crop_glob = f"outputs/plate_crops/Processed/*_track{track_id}_processed.jpg"
        matching_files = glob.glob(crop_glob)
        if matching_files:
            crop_path = matching_files[0]
            if os.path.exists(crop_path):
                try:
                    img_plate = OpenpyxlImage(crop_path)
                    img_plate.width = 100
                    img_plate.height = 35
                    ws.add_image(img_plate, f"H{row_idx}")
                except Exception:
                    ws.cell(row=row_idx, column=8, value="Img Err")
        else:
            ws.cell(row=row_idx, column=8, value="No Crop")

        # Locate and embed vehicle snapshot image
        if snapshot_path and os.path.exists(snapshot_path):
            try:
                img_vehicle = OpenpyxlImage(snapshot_path)
                # Fit 120x70 bounding box within the cell
                img_vehicle.width = 120
                img_vehicle.height = 70
                ws.add_image(img_vehicle, f"G{row_idx}")
            except Exception:
                ws.cell(row=row_idx, column=7, value="Img Err")
        else:
            ws.cell(row=row_idx, column=7, value="No Image")
            
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        if col_letter == "G":
            ws.column_dimensions["G"].width = 20
            continue
        if col_letter == "H":
            ws.column_dimensions["H"].width = 18
            continue
        for cell in col:
            val_str = str(cell.value or '')
            if len(val_str) > max_len:
                max_len = len(val_str)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 10)
        
    out_buf = BytesIO()
    wb.save(out_buf)
    out_buf.seek(0)
    return out_buf
