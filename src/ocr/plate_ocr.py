import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import cv2
import numpy as np
import easyocr
import torch
import math
import pytesseract

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9 ]')
MIN_PLATE_LENGTH = 4
MAX_PLATE_LENGTH = 10
TARGET_WIDTH = 300

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

def deskew_plate(binary):
    """
    Finds the largest contour and rotates the binary image to straighten the plate.
    Returns the straightened binary image.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
        
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:
        return binary

    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Avoid rotating for very small angles
    if abs(angle) < 1.0:
        return binary

    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # We pad the image so we don't cut off corners during rotation
    rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_plate_variant(cropped_plate_img, variant="adaptive"):
    """
    Prepares an image variant for OCR.
    Variants:
    - "grayscale": Resized, CLAHE enhanced, deskewed. Best for deep-learning EasyOCR.
    - "adaptive": Binarized using local adaptive thresholding (51px window).
    - "otsu": Binarized using global Otsu thresholding.
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    h, w = cropped_plate_img.shape[:2]
    if w == 0:
        return None
        
    # 1. Target-Width Scaling
    scale = TARGET_WIDTH / float(w)
    resized = cv2.resize(cropped_plate_img, (TARGET_WIDTH, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    if variant == "grayscale":
        # Deskew the grayscale image directly
        processed = deskew_plate(clahe)
        # Add white padding around the grayscale plate
        final_img = cv2.copyMakeBorder(processed, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
        return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

    # Apply Thresholding based on variant
    if variant == "otsu":
        _, binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Fixed Block Size: Increased block size from 11 to 51 to cover character stroke width
        binary = cv2.adaptiveThreshold(
            clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 51, 2
        )

    # Deskewing
    binary = deskew_plate(binary)

    # Background Color Verification
    bh, bw = binary.shape
    border_pixels = np.concatenate([
        binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
    ])
    white_pixels = np.count_nonzero(border_pixels == 255)
    black_pixels = len(border_pixels) - white_pixels
    
    if black_pixels > white_pixels:
        binary = cv2.bitwise_not(binary)

    # Morphological Closing
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Protective padding
    final_img = cv2.copyMakeBorder(cleaned, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def _run_ocr(processed, engine_name):
    """
    Runs the selected OCR engine and applies spatial sorting and text cleaning.
    """
    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    if engine_name == "PyTesseract":
        try:
            config = f"-c tessedit_char_whitelist={allowlist} --psm 7"
            text = pytesseract.image_to_string(processed, config=config)
            
            d = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
            
            confidences = []
            for c in d.get('conf', []):
                try:
                    val = int(float(c))
                    if val != -1:
                        confidences.append(val)
                except (ValueError, TypeError):
                    pass
            
            combined_text = PLATE_CHAR_PATTERN.sub('', text.upper())
            avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
        except Exception as e:
            print(f"[OCR] PyTesseract execution failed: {str(e)}")
            return '', 0.0
        
    else: # EasyOCR
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

        # Spatial sorting: bucket bounding boxes by roughly the same line (y / 20), then sort by x
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

    text = _fix_plate_format(combined_text)
    
    return text, adjusted_conf


def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """
    Lazy Multi-Variant OCR execution.
    """
    # 1. Try Grayscale first (Preserves gradients/edges for CRAFT/CRNN)
    processed_gray = preprocess_plate_variant(cropped_plate_img, "grayscale")
    if processed_gray is None:
        return '', 0.0, engine_name, None

    text, conf = _run_ocr(processed_gray, engine_name)
    if text and conf >= 0.40:
        return text, conf, engine_name, processed_gray

    # 2. Try Adaptive Threshold Fallback (Block size 51)
    processed_adaptive = preprocess_plate_variant(cropped_plate_img, "adaptive")
    text_adaptive, conf_adaptive = _run_ocr(processed_adaptive, engine_name)
    if text_adaptive and conf_adaptive >= 0.40:
        if conf_adaptive > conf:
            text, conf, processed_gray = text_adaptive, conf_adaptive, processed_adaptive
            
    # 3. Try Otsu Threshold Fallback
    processed_otsu = preprocess_plate_variant(cropped_plate_img, "otsu")
    text_otsu, conf_otsu = _run_ocr(processed_otsu, engine_name)

    # Return the best performing variant
    candidates = [
        (text, conf, processed_gray),
        (text_adaptive, conf_adaptive, processed_adaptive),
        (text_otsu, conf_otsu, processed_otsu)
    ]
    best_text, best_conf, best_processed = max(candidates, key=lambda c: c[1])
    return best_text, best_conf, engine_name, best_processed
