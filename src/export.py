import glob
import os
from io import BytesIO
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

def build_xlsx_report(detections_df: pd.DataFrame) -> BytesIO:
    """
    Generates a stylized Excel report with embedded crop images of vehicles and license plates.
    
    Returns a BytesIO binary buffer containing the spreadsheet so the web dashboard can stream it directly.
    """
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "ALPR Detections"
    
    # 1. Define columns headers. Columns G and H are reserved for rendering embedded images.
    headers = ["Track ID", "Timestamp", "Vehicle Type", "Color", "Plate Number", "Confidence", "Vehicle Image", "Plate Crop"]
    worksheet.append(headers)
    
    # 2. Styling the header row to look clean and professional (white text on a deep blue background)
    # Color hex '1F497D' is a professional dark steel blue
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center")
    
    # Apply the styling to all cells in the first row
    for column_index in range(1, len(headers) + 1):
        cell = worksheet.cell(row=1, column=column_index)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        
    # Set header row height to 25px to give it some comfortable breathing room
    worksheet.row_dimensions[1].height = 25
    
    # Stylings for standard data cells
    font_regular = Font(name="Arial", size=10)
    align_center = Alignment(horizontal="center", vertical="center")
    
    # 3. Populate rows with vehicle telemetry data
    for idx, row in detections_df.iterrows():
        # Excel is 1-indexed, and row 1 is the header. So data begins at index + 2.
        row_index = idx + 2
        
        track_id = row.get("track_id", "")
        timestamp = row.get("timestamp", "")
        vehicle_type = row.get("vehicle_type", "")
        color = row.get("color", "")
        plate_number = row.get("plate_number", "")
        confidence = row.get("confidence", 0.0)
        snapshot_path = row.get("snapshot_path", "")
        
        # Populate text cells, centering values
        worksheet.cell(row=row_index, column=1, value=int(track_id) if str(track_id).isdigit() else track_id)
        worksheet.cell(row=row_index, column=2, value=str(timestamp))
        worksheet.cell(row=row_index, column=3, value=str(vehicle_type).capitalize())
        worksheet.cell(row=row_index, column=4, value=str(color).capitalize() if pd.notna(color) else "")
        worksheet.cell(row=row_index, column=5, value=str(plate_number))
        
        # Format confidence cell to display as a percentage (e.g. 0.825 becomes "83%")
        confidence_value = float(confidence) if pd.notna(confidence) else 0.0
        confidence_cell = worksheet.cell(row=row_index, column=6, value=confidence_value)
        confidence_cell.number_format = "0%"
        
        # Apply standard font styling to the text columns (columns 1 to 6)
        for column_index in range(1, 7):
            cell = worksheet.cell(row=row_index, column=column_index)
            cell.font = font_regular
            cell.alignment = align_center
        
        # Set height of each data row to 80px to accommodate the vehicle snapshots and plate crops
        worksheet.row_dimensions[row_index].height = 80
        
        # 4. Locate and embed corresponding processed plate crop for this track
        crop_glob = f"outputs/plate_crops/Processed/*_track{track_id}_processed.jpg"
        matching_files = glob.glob(crop_glob)
        if matching_files:
            crop_path = matching_files[0]
            if os.path.exists(crop_path):
                try:
                    img_plate = OpenpyxlImage(crop_path)
                    # Constrain image size to 100x35 pixels so it fits cleanly in column H
                    img_plate.width = 100
                    img_plate.height = 35
                    worksheet.add_image(img_plate, f"H{row_index}")
                except Exception:
                    worksheet.cell(row=row_index, column=8, value="Img Err")
        else:
            worksheet.cell(row=row_index, column=8, value="No Crop")
 
        # 5. Locate and embed full vehicle context snapshot image
        if snapshot_path and os.path.exists(snapshot_path):
            try:
                img_vehicle = OpenpyxlImage(snapshot_path)
                # Constrain image size to 120x70 pixels so it fits cleanly inside column G
                img_vehicle.width = 120
                img_vehicle.height = 70
                worksheet.add_image(img_vehicle, f"G{row_index}")
            except Exception:
                worksheet.cell(row=row_index, column=7, value="Img Err")
        else:
            worksheet.cell(row=row_index, column=7, value="No Image")
            
    # 6. Auto-fit column widths based on text length to prevent text clipping
    for column in worksheet.columns:
        max_len = 0
        column_letter = get_column_letter(column[0].column)
        
        # Column G is reserved for the vehicle image; keep width at 20
        if column_letter == "G":
            worksheet.column_dimensions["G"].width = 20
            continue
        # Column H is reserved for the plate crop image; keep width at 18
        if column_letter == "H":
            worksheet.column_dimensions["H"].width = 18
            continue
            
        # For all text columns, find the longest string value
        for cell in column:
            value_string = str(cell.value or '')
            if len(value_string) > max_len:
                max_len = len(value_string)
        # Add 3 characters padding; set a minimum default width of 10
        worksheet.column_dimensions[column_letter].width = max(max_len + 3, 10)
        
    # Save the Excel file in-memory and rewind the stream position to the beginning
    out_buf = BytesIO()
    workbook.save(out_buf)
    out_buf.seek(0)
    return out_buf
