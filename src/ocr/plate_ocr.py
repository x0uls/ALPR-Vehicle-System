import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import cv2
import numpy as np
import easyocr
import torch
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
    """Fix common OCR misreads based on Malaysian plate format by searching partitions."""
    compact = text.replace(' ', '').upper()
    if len(compact) < MIN_PLATE_LENGTH:
        return text

    digit_to_letter = str.maketrans('01258', 'OIZSB')
    letter_to_digit = str.maketrans('OIZSBGDTQ', '012586007')

    special_prefixes = ["PUTRAJAYA", "RIMAU", "1M4U", "PERODUA", "PROTON"]
    
    prefix_lengths = [1, 2, 3]
    for sp in special_prefixes:
        if compact.startswith(sp):
            prefix_lengths = [len(sp)]
            break

    best_corrected = None
    best_changes = 999

    for p_len in prefix_lengths:
        for has_suffix in [False, True]:
            if has_suffix:
                if len(compact) <= p_len + 1:
                    continue
                pref = compact[:p_len]
                mid = compact[p_len:-1]
                suff = compact[-1]
            else:
                if len(compact) <= p_len:
                    continue
                pref = compact[:p_len]
                mid = compact[p_len:]
                suff = ""

            if not (1 <= len(mid) <= 4):
                continue

            pref_corr = pref.translate(digit_to_letter)
            mid_corr = mid.translate(letter_to_digit)
            suff_corr = suff.translate(digit_to_letter) if suff else ""

            is_valid_pref = (pref_corr in special_prefixes) or (re.match(r'^[A-Z]{1,3}$', pref_corr) is not None)
            is_valid_mid = (re.match(r'^\d{1,4}$', mid_corr) is not None)
            is_valid_suff = (not suff_corr) or (re.match(r'^[A-Z]$', suff_corr) is not None)

            if is_valid_pref and is_valid_mid and is_valid_suff:
                changes = (
                    sum(1 for c1, c2 in zip(pref, pref_corr) if c1 != c2) +
                    sum(1 for c1, c2 in zip(mid, mid_corr) if c1 != c2) +
                    sum(1 for c1, c2 in zip(suff, suff_corr) if c1 != c2)
                )
                if changes < best_changes:
                    best_changes = changes
                    best_corrected = pref_corr + " " + mid_corr + suff_corr

    if best_corrected:
        return best_corrected
    return text

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

    h, w = cropped_plate_img.shape[:2]
    aspect = w / h
    resized = cv2.resize(cropped_plate_img, (TARGET_WIDTH, int(TARGET_WIDTH / aspect)))
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
    target_h = 150
    h, w = cropped_plate_img.shape[:2]
    aspect = w / h
    resized = cv2.resize(cropped_plate_img, (int(target_h * aspect), target_h))
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

    # ─── Step 2: Deskew Grayscale ───
    deskewed_gray = auto_deskew(gray)

    # ─── Step 3: Bilateral Filter ───
    filtered = cv2.bilateralFilter(deskewed_gray, 9, 75, 75)

    # ─── Step 4: Polarity Detection & Inversion ───
    _, test_bin = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(test_bin) < 127:
        filtered = ~filtered

    # ─── Step 5: Binarization ───
    if threshold_method == "otsu":
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 2
        )

    # ─── Step 5b: Morphological noise cleanup ───
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # ─── Step 6: Border Removal & Cropping ───
    inv_binary = ~binary
    padded_inv = cv2.copyMakeBorder(inv_binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded_inv.shape[:2]

    contours, _ = cv2.findContours(padded_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    char_bboxes = [cv2.boundingRect(c) for c in contours]
    valid_bboxes = [
        (x, y, w, h) for (x, y, w, h) in char_bboxes
        if w >= 2 and h >= 8 and w < 0.85 * pw and h < 0.85 * ph
    ]

    if valid_bboxes:
        x_min = min(b[0] for b in valid_bboxes)
        y_min = min(b[1] for b in valid_bboxes)
        x_max = max(b[0] + b[2] for b in valid_bboxes)
        y_max = max(b[1] + b[3] for b in valid_bboxes)
        
        margin = 4
        x1, y1 = max(0, x_min - margin), max(0, y_min - margin)
        x2, y2 = min(pw, x_max + margin), min(ph, y_max + margin)
        cropped_padded = padded_inv[y1:y2, x1:x2]
    else:
        cropped_padded = inv_binary

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
            config = f"--psm 7 --oem 1"
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
