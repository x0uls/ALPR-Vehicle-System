import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import cv2
import numpy as np
import easyocr
import torch
import math
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

def deskew_plate(img):
    """Safely finds the largest contour and straightens the plate,
       handling both binary and grayscale inputs smoothly."""
    # If the input image is grayscale/CLAHE, create a temporary binary mask for contour tracking
    if len(img.shape) == 2:
        unique_vals = len(np.unique(img))
        if unique_vals > 2:  # It's grayscale, not binary
            # Use a quick Otsu threshold to separate the text/frame out for alignment
            _, binary_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            binary_mask = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
        
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:
        return img

    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Handle rotation angles across different OpenCV version specifications cleanly
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 1.0:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_for_easyocr(cropped_plate_img):
    """Preprocessing pipeline optimized for EasyOCR: Grayscale, CLAHE, deskew."""
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    h, w = cropped_plate_img.shape[:2]
    if w == 0 or h == 0:
        return None

    scale = TARGET_WIDTH / float(w)
    resized = cv2.resize(cropped_plate_img, (TARGET_WIDTH, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    processed = deskew_plate(clahe)
    
    final_img = cv2.copyMakeBorder(processed, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
def preprocess_for_tesseract(cropped_plate_img, threshold_method="adaptive"):
    """Clean-room preprocessing pipeline for PyTesseract.
    
    Steps:
    1. Grayscale conversion
    2. Bilateral filtering (denoise, preserve edges)
    3. Adaptive thresholding (handles uneven lighting, avoids border capture)
    4. Shrink-crop strategy (contour-based border trimming)
    5. clear_border (remove edge-touching artifacts)
    6. DPI upscale (2x if crop is small)
    7. Deskew, polarity fix, white padding
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    h, w = cropped_plate_img.shape[:2]
    if w == 0 or h == 0:
        return None

    # ─── Step 1: Grayscale ───
    gray = cv2.cvtColor(cropped_plate_img, cv2.COLOR_BGR2GRAY)

    # ─── Step 2: Bilateral Filter (denoise, keep edges sharp) ───
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # ─── Step 3: Adaptive Thresholding ───
    # Adaptive handles uneven lighting across the plate and avoids
    # capturing the black mounting frame as text (a known Otsu failure mode).
    if threshold_method == "otsu":
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 2
        )

    # ─── Step 4: Corrected Shrink-Crop Strategy ───
    # We must invert the image temporarily so that the black frame and text
    # become white foreground structures (255) for findContours to locate.
    inv_for_contours = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inv_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        all_points = np.vstack(contours)
        x, y, cw, ch = cv2.boundingRect(all_points)
        margin = 4
        sx = min(x + margin, binary.shape[1] - 1)
        sy = min(y + margin, binary.shape[0] - 1)
        ex = max(x + cw - margin, sx + 1)
        ey = max(y + ch - margin, sy + 1)
        binary = binary[sy:ey, sx:ex]

    # ─── Step 5: clear_border ───
    # Remove any noise or dark lines still touching the outer edges.
    # Invert so border artifacts are foreground (nonzero), clear them, invert back.
    inverted = cv2.bitwise_not(binary)
    cleared = clear_border(inverted).astype(np.uint8)
    binary = cv2.bitwise_not(cleared * 255)

    # ─── Step 6: DPI Upscale ───
    # Tesseract performs best at ~300 DPI. Small crops need upscaling.
    bh, bw = binary.shape[:2]
    if bh < 100:
        scale = 2.0
        binary = cv2.resize(binary, (int(bw * scale), int(bh * scale)), interpolation=cv2.INTER_CUBIC)

    # ─── Step 7: Deskew ───
    binary = deskew_plate(binary)

    # ─── Step 8: Polarity Fix (ensure black text on white background) ───
    total_pixels = binary.size
    black_count = total_pixels - cv2.countNonZero(binary)
    if black_count > total_pixels * 0.5:
        binary = cv2.bitwise_not(binary)

    # ─── Step 9: White Padding ───
    # Add generous white border so characters don't touch the image edges.
    final_img = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    # PyTesseract natively prefers single-channel grayscale arrays.
    # Returning a single channel image prevents internal re-conversion overhead.
    return final_img


def _run_ocr(processed, engine_name):
    """Runs the selected OCR engine and applies spatial sorting and text cleaning."""
    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    if engine_name == "PyTesseract":
        try:
            config = f"-c tessedit_char_whitelist={allowlist} --psm 7 --oem 1"
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
        # Pass 1: Try Grayscale first (Best for raw LSTM gradients)
        processed_gray = preprocess_for_easyocr(cropped_plate_img)
        if processed_gray is not None:
            text_gray, conf_gray = _run_ocr(processed_gray, engine_name)
            if text_gray and conf_gray >= 0.50:
                return text_gray, conf_gray, engine_name, processed_gray
        else:
            text_gray, conf_gray = '', 0.0

        # Pass 2: Try Otsu binarization
        processed_otsu = preprocess_for_tesseract(cropped_plate_img, "otsu")
        if processed_otsu is not None:
            text_otsu, conf_otsu = _run_ocr(processed_otsu, engine_name)
            if text_otsu and conf_otsu >= 0.50:
                return text_otsu, conf_otsu, engine_name, processed_otsu
        else:
            text_otsu, conf_otsu = '', 0.0

        # Pass 3: Try Adaptive Gaussian thresholding
        processed_adapt = preprocess_for_tesseract(cropped_plate_img, "adaptive")
        if processed_adapt is not None:
            text_adapt, conf_adapt = _run_ocr(processed_adapt, engine_name)
            if text_adapt and conf_adapt >= 0.50:
                return text_adapt, conf_adapt, engine_name, processed_adapt
        else:
            text_adapt, conf_adapt = '', 0.0

        # FIX: Complete the fallback strategy if no image variant beats the 0.50 threshold.
        # Pick the variant that yielded text and has the highest confidence.
        candidates = [
            (text_gray, conf_gray, processed_gray),
            (text_otsu, conf_otsu, processed_otsu),
            (text_adapt, conf_adapt, processed_adapt)
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
