import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import cv2
import numpy as np
import easyocr
import torch
import imutils
from deskew import determine_skew
from skimage.transform import rotate
from skimage.segmentation import clear_border
import pytesseract

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9 ]')
MIN_PLATE_LENGTH = 4
MAX_PLATE_LENGTH = 10
TARGET_WIDTH = 300

# Malaysian plate format: prefix (1-3 letters or special words) + 1-4 digits + optional trailing letter
MALAYSIAN_PLATE_REGEX = re.compile(
    r'^(PUTRAJAYA|RIMAU|1M4U|PERODUA|PROTON|[A-Z]{1,3})\s?\d{1,4}(\s?[A-Z])?$'
)

def _fix_plate_format(text):
    """Fix common OCR misreads based on Malaysian plate format."""
    compact = text.replace(' ', '')
    if len(compact) < MIN_PLATE_LENGTH:
        return text

    digit_to_letter = str.maketrans('01258', 'OIZSB')
    letter_to_digit = str.maketrans('OIZSB', '01258')
    extra_letter_to_digit = {'G': '6', 'D': '0', 'Q': '0', 'T': '7'}

    first_digit_pos = None
    for i, ch in enumerate(compact):
        if ch.isdigit():
            first_digit_pos = i
            break

    if first_digit_pos is None:
        return text

    last_digit_pos = first_digit_pos
    for i in range(first_digit_pos, len(compact)):
        if compact[i].isdigit():
            last_digit_pos = i

    result = []
    for i, ch in enumerate(compact):
        if i < first_digit_pos:
            if ch.isdigit():
                ch = ch.translate(digit_to_letter)
            result.append(ch)
        elif i <= last_digit_pos:
            if ch.isalpha():
                if ch in extra_letter_to_digit:
                    ch = extra_letter_to_digit[ch]
                else:
                    ch = ch.translate(letter_to_digit)
            result.append(ch)
        else:
            if ch.isdigit():
                ch = ch.translate(digit_to_letter)
            result.append(ch)

    return ''.join(result)

def auto_deskew(img):
    """Replaces manual contour and matrix rotation math with Hough transforms."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    angle = determine_skew(gray)
    
    if angle is None or abs(angle) < 1.0:
        return img
        
    rotated = rotate(img, angle, resize=True, mode='edge')
    return (rotated * 255).astype(np.uint8)

def preprocess_for_easyocr(cropped_plate_img):
    """Preprocessing pipeline optimized for EasyOCR: Grayscale, CLAHE, deskew."""
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    resized = imutils.resize(cropped_plate_img, width=TARGET_WIDTH)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    processed = auto_deskew(clahe)
    
    final_img = cv2.copyMakeBorder(processed, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

def preprocess_for_tesseract(cropped_plate_img, threshold_method="adaptive"):
    """Enhanced preprocessing pipeline for PyTesseract.
    
    Ensures black text on a white background and removes the carplate border/outline.
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    # ─── Step 1: DPI Upscale and Grayscale ───
    # Tesseract prefers images where character height is at least 30-50 pixels.
    # We resize to a standard target height (150px) using imutils.
    target_h = 150
    resized = imutils.resize(cropped_plate_img, height=target_h)
    
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # ─── Step 2: Deskew Grayscale ───
    # Deskewing works best on the raw grayscale image before border removal.
    deskewed_gray = auto_deskew(gray)

    # ─── Step 3: Bilateral Filter (denoise, keep edges sharp) ───
    filtered = cv2.bilateralFilter(deskewed_gray, 9, 75, 75)

    # ─── Step 4: Polarity Detection & Inversion ───
    # Measure average intensity of pixels near the image boundaries.
    # If the border region is dark (intensity < 120), it's a light-on-dark plate.
    # We invert it before thresholding so it is treated as dark-on-light (black on white).
    bh_f, bw_f = filtered.shape[:2]
    border_mask = np.zeros_like(filtered, dtype=np.uint8)
    border_width = max(1, int(min(bh_f, bw_f) * 0.08))  # 8% border width
    border_mask[:border_width, :] = 255
    border_mask[-border_width:, :] = 255
    border_mask[:, :border_width] = 255
    border_mask[:, -border_width:] = 255
    avg_border = cv2.mean(filtered, mask=border_mask)[0]

    is_light_on_dark = avg_border < 120
    if is_light_on_dark:
        filtered = cv2.bitwise_not(filtered)

    # ─── Step 5: Binarization (results in white background [255], black text/borders [0]) ───
    if threshold_method == "otsu":
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 2
        )

    # ─── Step 6: Border Removal & Cropping via Character Contour Extraction ───
    # Invert binary image so foreground (characters and borders) is white (255) on black (0)
    inv_binary = cv2.bitwise_not(binary)

    # Pad with 10px of black background so characters close to edges don't touch padded border
    padded_inv = cv2.copyMakeBorder(inv_binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded_inv.shape[:2]

    # Find contours in the padded image (use RETR_LIST to find nested characters inside the outer border)
    contours, _ = cv2.findContours(padded_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    char_contours = []
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        
        # Filter out noise (too small)
        if w_c < 2 or h_c < 8:
            continue
            
        # Filter out borders (which span a massive portion of the plate)
        if w_c > 0.85 * pw or h_c > 0.85 * ph:
            continue
            
        # Keep character candidates (even thin ones like '1' / 'I' and small suffix letters)
        # Using a conservative contour area check
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < 8:
            continue
            
        char_contours.append(cnt)

    if char_contours:
        all_pts = np.vstack(char_contours)
        x_min, y_min, w_crop, h_crop = cv2.boundingRect(all_pts)
        
        # Crop the padded image to the character bounding box with a small margin
        margin = 4
        x1 = max(0, x_min - margin)
        y1 = max(0, y_min - margin)
        x2 = min(pw, x_min + w_crop + margin)
        y2 = min(ph, y_min + h_crop + margin)
        cropped_padded = padded_inv[y1:y2, x1:x2]
    else:
        cropped_padded = inv_binary

    # Invert back to black text on white background
    final_binary = cv2.bitwise_not(cropped_padded)

    # ─── Step 7: Final White Padding ───
    # Add generous white padding so characters don't touch edges (Tesseract requirement)
    padded_final = cv2.copyMakeBorder(final_binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    
    return padded_final


def _run_ocr(processed, engine_name):
    """Runs the selected OCR engine and applies spatial sorting and text cleaning."""
    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    if engine_name == "PyTesseract":
        try:
            config = "--psm 7 --oem 1"
            d = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
            
            words = []
            confidences = []
            
            for i in range(len(d.get('text', []))):
                word = str(d['text'][i]).strip()
                cleaned_word = PLATE_CHAR_PATTERN.sub('', word.upper())
                
                # Skip layout tokens that contain no usable alphanumeric info
                if not cleaned_word:
                    continue
                    
                conf_val = d['conf'][i]
                try:
                    val = int(float(conf_val))
                except (ValueError, TypeError):
                    val = -1
                
                # FIX: If clean text was read but Tesseract returned -1 layout confidence,
                # assign a 50% baseline score so the valid read isn't thrown away.
                if val == -1:
                    val = 50 
                    
                words.append(cleaned_word)
                confidences.append(val)
            
            combined_text = " ".join(words)
            avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
            print(f"[OCR DEBUG] PyTesseract raw='{' '.join(d.get('text', []))}', cleaned='{combined_text}', conf={avg_conf:.3f}")
        except Exception as e:
            print(f"[OCR] PyTesseract execution failed: {str(e)}")
            return '', 0.0
        
    else: # EasyOCR Flow
        results = reader.readtext(
            processed, 
            allowlist=allowlist, 
            paragraph=False,
            text_threshold=0.5, 
            low_text=0.3, 
            mag_ratio=1.0
        )

        if not results:
            return '', 0.0

        results_sorted = sorted(results, key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
        raw_texts = [r[1].upper().strip() for r in results_sorted]
        confidences = [float(r[2]) for r in results_sorted if len(r) > 2]
        
        combined_text = " ".join(raw_texts)
        combined_text = PLATE_CHAR_PATTERN.sub('', combined_text)
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

    combined_text = ' '.join(combined_text.split())
    compact = combined_text.replace(' ', '')
    
    if len(compact) < MIN_PLATE_LENGTH or len(compact) > MAX_PLATE_LENGTH:
        return '', 0.0

    # Length-Weighted Confidence
    adjusted_conf = avg_conf * min(1.0, len(compact) / 7.0)
    
    if combined_text.strip():
        adjusted_conf = max(0.01, adjusted_conf)

    text = _fix_plate_format(combined_text)

    # Format validation: boost if valid, penalize if structural garbage
    compact_text = text.replace(' ', '')
    if MALAYSIAN_PLATE_REGEX.match(compact_text):
        adjusted_conf = min(1.0, adjusted_conf * 1.15)
    else:
        # Drop the score of non-standard formats so they lose out in the fallback selection
        adjusted_conf *= 0.5

    return text, adjusted_conf

def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """Lazy Multi-Variant OCR execution wrapper."""
    if engine_name == "PyTesseract":
        # Pass 1: Try Adaptive Gaussian thresholding (Best for typical lighting variation)
        processed_adapt = preprocess_for_tesseract(cropped_plate_img, "adaptive")
        if processed_adapt is not None:
            text_adapt, conf_adapt = _run_ocr(processed_adapt, engine_name)
            if text_adapt and conf_adapt >= 0.50:
                return text_adapt, conf_adapt, engine_name, processed_adapt
        else:
            text_adapt, conf_adapt = '', 0.0

        # Pass 2: Try Otsu binarization
        processed_otsu = preprocess_for_tesseract(cropped_plate_img, "otsu")
        if processed_otsu is not None:
            text_otsu, conf_otsu = _run_ocr(processed_otsu, engine_name)
            if text_otsu and conf_otsu >= 0.50:
                return text_otsu, conf_otsu, engine_name, processed_otsu
        else:
            text_otsu, conf_otsu = '', 0.0

        # Pass 3: Try Grayscale CLAHE (as final fallback)
        processed_gray = preprocess_for_easyocr(cropped_plate_img)
        if processed_gray is not None:
            text_gray, conf_gray = _run_ocr(processed_gray, engine_name)
            if text_gray and conf_gray >= 0.50:
                return text_gray, conf_gray, engine_name, processed_gray
        else:
            text_gray, conf_gray = '', 0.0

        # Pick the variant that yielded text and has the highest confidence.
        candidates = [
            (text_adapt, conf_adapt, processed_adapt),
            (text_otsu, conf_otsu, processed_otsu),
            (text_gray, conf_gray, processed_gray)
        ]
        valid_candidates = [c for c in candidates if c[0].strip()]
        if valid_candidates:
            best_candidate = max(valid_candidates, key=lambda x: x[1])
            return best_candidate[0], best_candidate[1], engine_name, best_candidate[2]
            
        return '', 0.0, engine_name, None

    else:
        # Standard Engine execution flow (EasyOCR)
        processed = preprocess_for_easyocr(cropped_plate_img)
        if processed is None:
            return '', 0.0, engine_name, None
        text, conf = _run_ocr(processed, engine_name)
        return text, conf, engine_name, processed
