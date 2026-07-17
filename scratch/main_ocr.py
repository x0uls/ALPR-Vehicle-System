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

    corrected = ''.join(result)

    # Re-insert a single space at the letter→digit boundary for readability
    for i in range(1, len(corrected)):
        if corrected[i-1].isalpha() and corrected[i].isdigit():
            return corrected[:i] + ' ' + corrected[i:]
    return corrected

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
    # Perform a quick Otsu thresholding to count black vs white pixels.
    # The plate background dominates the crop area. If the majority of pixels are black (0),
    # it is a dark background plate. We invert it to black-on-white.
    _, test_bin = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(test_bin == 0) > np.sum(test_bin == 255):
        filtered = cv2.bitwise_not(filtered)

    # ─── Step 5: Binarization (results in white background [255], black text/borders [0]) ───
    if threshold_method == "otsu":
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 2
        )

    # ─── Step 5b: Morphological noise cleanup ───
    # Remove salt-and-pepper dots that Tesseract misreads as punctuation.
    # Applied after polarity+binarization so text is black(0) on white(255).
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel, iterations=1)

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
    
    padded_final = cv2.GaussianBlur(padded_final, (3, 3), 0)

    return padded_final


def _run_ocr(processed, engine_name):
    """Runs the selected OCR engine and applies spatial sorting and text cleaning."""
    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    if engine_name == "PyTesseract":
        try:
            config = f"--psm 7 --oem 1 -c tessedit_char_whitelist={allowlist}"
            d = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
            
            words_confs = []
            for w, c in zip(d.get('text', []), d.get('conf', [])):
                cleaned = PLATE_CHAR_PATTERN.sub('', str(w).strip().upper())
                if cleaned:
                    try:
                        val = int(float(c))
                    except (ValueError, TypeError):
                        val = -1
                    words_confs.append((cleaned, 50 if val == -1 else val))
            
            words = [wc[0] for wc in words_confs]
            confidences = [wc[1] for wc in words_confs]
            combined_text = " ".join(words)
            avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
            print(f"[OCR DEBUG] PyTesseract raw='{' '.join(d.get('text', []))}', cleaned='{combined_text}', conf={avg_conf:.3f}")
        except Exception as e:
            print(f"[OCR] PyTesseract execution failed: {str(e)}")
            return '', 0.0
        
    else: # EasyOCR Flow
        results = reader.readtext(processed, allowlist=allowlist, paragraph=False, text_threshold=0.3, low_text=0.2, mag_ratio=1.5)
        if not results:
            return '', 0.0

        results_sorted = sorted(results, key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
        combined_text = PLATE_CHAR_PATTERN.sub('', " ".join([r[1].upper().strip() for r in results_sorted]))
        confidences = [float(r[2]) for r in results_sorted if len(r) > 2]
        avg_conf = np.mean(confidences) if confidences else 0.0

    compact = ' '.join(combined_text.split()).replace(' ', '')
    if len(compact) < MIN_PLATE_LENGTH or len(compact) > MAX_PLATE_LENGTH:
        return '', 0.0

    adjusted_conf = max(0.01, avg_conf * min(1.0, len(compact) / 7.0)) if combined_text.strip() else 0.0
    text = _fix_plate_format(combined_text)

    # Strictly enforce Malaysian plate format (prefix letters + digits + optional suffix letter)
    if not MALAYSIAN_PLATE_REGEX.match(text.replace(' ', '')):
        return '', 0.0

    adjusted_conf = min(1.0, adjusted_conf * 1.15)
    return text, adjusted_conf

def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """Lazy Multi-Variant OCR execution wrapper."""
    if engine_name != "PyTesseract":
        processed = preprocess_for_easyocr(cropped_plate_img)
        if processed is None:
            return '', 0.0, engine_name, None
        text, conf = _run_ocr(processed, engine_name)
        return text, conf, engine_name, processed

    passes = [
        (preprocess_for_tesseract, "adaptive"),
        (preprocess_for_tesseract, "otsu"),
    ]
    candidates = []
    for prep_fn, thresh in passes:
        processed = prep_fn(cropped_plate_img) if thresh is None else prep_fn(cropped_plate_img, thresh)
        if processed is not None:
            text, conf = _run_ocr(processed, engine_name)
            if text and conf >= 0.50:
                return text, conf, engine_name, processed
            candidates.append((text, conf, processed))

    valid = [c for c in candidates if c[0].strip()]
    if valid:
        best = max(valid, key=lambda x: x[1])
        return best[0], best[1], engine_name, best[2]
        
    return '', 0.0, engine_name, None
