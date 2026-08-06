import os
# Limit OpenMP threads to 1 to prevent CPU thrashing/bottlenecks during inference
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import cv2
import numpy as np
import easyocr
import torch
import pytesseract

# Auto-detect Tesseract executable path across Linux / Colab and Windows
for t_path in ["/usr/bin/tesseract", "/usr/local/bin/tesseract", r"C:\Program Files\Tesseract-OCR\tesseract.exe"]:
    if os.path.exists(t_path):
        pytesseract.pytesseract.tesseract_cmd = t_path
        break

# Initialize EasyOCR reader at import time (thread-safe for downstream worker pools)
# ['en']: English character dictionary; gpu: CUDA hardware acceleration
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Regex to strip non-alphanumeric characters
PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9 ]')
MIN_PLATE_LENGTH = 4
MAX_PLATE_LENGTH = 10
TARGET_WIDTH = 300

SPECIAL_PLATE_PREFIXES = ["PUTRAJAYA", "RIMAU", "1M4U", "PERODUA", "PROTON"]
MALAYSIAN_PLATE_REGEX = re.compile(
    r'^(' + '|'.join(SPECIAL_PLATE_PREFIXES) + r'|[A-Z]{1,3})\s?\d{1,4}(\s?[A-Z])?$'
)

def _resize_keep_aspect(img, target, by="width"):
    """Resizes an image while preserving aspect ratio using OpenCV."""
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img
    if by == "width":
        if w == target:
            return img
        new_h = max(1, int(h * (target / float(w))))
        return cv2.resize(img, (target, new_h), interpolation=cv2.INTER_AREA)
    else:
        if h == target:
            return img
        new_w = max(1, int(w * (target / float(h))))
        return cv2.resize(img, (new_w, target), interpolation=cv2.INTER_AREA)


def _fix_plate_format(text):
    """Corrects common OCR letter/digit confusion based on Malaysian license plate formats."""
    compact_text = text.replace(' ', '').upper()
    if len(compact_text) < MIN_PLATE_LENGTH:
        return text

    digit_to_letter = str.maketrans('01258', 'OIZSB')
    letter_to_digit = str.maketrans('OIZSB', '01258')
    extra_letter_to_digit = {'G': '6', 'D': '0', 'Q': '0', 'T': '7'}

    # First-character interception: Malaysian plates ALWAYS start with a letter.
    # If OCR hallucinated the first character as a digit (e.g. 8QK→BQK, 0PP→OPP),
    # force-convert it to a letter BEFORE calculating first_digit_index.
    # Without this, the function would treat the hallucinated digit as the start
    # of the number block and refuse to correct it back.
    first_char = compact_text[0]
    if first_char.isdigit():
        compact_text = first_char.translate(digit_to_letter) + compact_text[1:]

    first_digit_index = next((i for i, char in enumerate(compact_text) if char.isdigit()), None)
    if first_digit_index is None:
        return text

    last_digit_index = max(i for i, char in enumerate(compact_text) if char.isdigit())

    corrected_chars = []
    for i, char in enumerate(compact_text):
        if i < first_digit_index:
            corrected_chars.append(char.translate(digit_to_letter) if char.isdigit() else char)
        elif i <= last_digit_index:
            if char.isalpha():
                char = extra_letter_to_digit.get(char, char.translate(letter_to_digit))
            corrected_chars.append(char)
        else:
            corrected_chars.append(char.translate(digit_to_letter) if char.isdigit() else char)

    corrected_text = ''.join(corrected_chars)
    for i in range(1, len(corrected_text)):
        if corrected_text[i-1].isalpha() and corrected_text[i].isdigit():
            return corrected_text[:i] + ' ' + corrected_text[i:]
    return corrected_text


# ─── Illumination & Preprocessing Helpers ─────────────────────────

def _apply_adaptive_gamma(gray):
    """
    Adjusts image gamma based on mean pixel intensity with 4 bands to handle
    a wider range of lighting conditions than the original 2-band version.
    - Very dark  (mean < 50):  aggressive brightening  (gamma 0.45)
    - Dark       (mean < 90):  moderate brightening     (gamma 0.65)
    - Normal     (90-170):     no change
    - Bright     (mean > 170): moderate darkening        (gamma 1.4)
    - Very bright(mean > 210): aggressive darkening      (gamma 1.8)
    """
    mean_val = np.mean(gray)
    if mean_val < 50:
        gamma = 0.45
    elif mean_val < 90:
        gamma = 0.65
    elif mean_val > 210:
        gamma = 1.8
    elif mean_val > 170:
        gamma = 1.4
    else:
        return gray

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray, table)


def _normalize_illumination_dog(gray):
    """
    Difference-of-Gaussians (DoG) illumination normalization.
    Subtracts a heavily blurred version from the image to remove low-frequency
    lighting gradients (e.g. half the plate in shadow, half in sunlight), then
    rescales to full 0-255 range. Very effective for uneven / partial lighting.
    """
    blur_small = cv2.GaussianBlur(gray, (3, 3), 1)
    blur_large = cv2.GaussianBlur(gray, (51, 51), 20)
    # Subtract the low-frequency illumination and shift to unsigned range
    dog = cv2.subtract(blur_small, blur_large)
    # Normalize to full 0-255 range
    return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _clear_borders(gray, margin_pct=0.05):
    """
    Crops inward by a percentage to shave off the thick black plastic frames
    that surround physical license plates. These frames confuse OCR engines
    into hallucinating extra characters.
    """
    h, w = gray.shape[:2]
    my = int(h * margin_pct)
    mx = int(w * margin_pct)
    # Ensure we don't crop to nothing
    if h - 2 * my < 10 or w - 2 * mx < 20:
        return gray
    return gray[my:h - my, mx:w - mx]


def _deskew_plate(gray):
    """
    Detects plate skew angle using Hough lines and corrects rotation up to ±15°.
    Skewed plates are a common OCR failure mode (characters get distorted).
    Returns the deskewed grayscale image, or the original if no strong lines found.
    """
    h, w = gray.shape[:2]
    if h < 10 or w < 20:
        return gray

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=w // 4, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, dx))
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return gray

    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=int(np.median(gray)))
    return rotated


def _upscale_if_small(gray, min_height=40):
    """Upscales small plate crops so OCR engines have enough pixel detail."""
    h, w = gray.shape[:2]
    if h < min_height:
        scale = min_height / float(h)
        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def _stretch_histogram(gray):
    """
    Stretches the pixel intensity range to use the full 0-255 range.
    Helps when the crop has very low dynamic range (fog, haze, washed-out).
    """
    pmin, pmax = np.percentile(gray, (2, 98))
    if pmax - pmin < 30:
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.clip((gray.astype(np.float32) - pmin) * 255.0 / max(1, pmax - pmin), 0, 255).astype(np.uint8)


def _auto_invert(binary):
    """
    Auto-invert check: if the majority of pixels are dark, the image likely has
    white text on dark background. Invert it so PyTesseract gets black-on-white.
    Uses median (more robust to noise than mean of border pixels).
    """
    if np.median(binary) < 127:
        return cv2.bitwise_not(binary)
    return binary


def _add_white_padding(img, pad=15):
    """Adds white border padding around the image for better OCR boundary detection."""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


# ─── Multi-Variant Preprocessing Strategies ───────────────────────
#
# PyTesseract expects clean binarized document scans (black text on white bg).
# EasyOCR uses a neural network that needs natural grayscale gradients preserved.
# Each engine gets 3 strategies optimized for different lighting conditions,
# and the best OCR result across all variants is selected.

# === PYTESSERACT STRATEGIES ===

def _preprocess_tophat_tess(gray):
    """
    Top-Hat Transform + Adaptive Thresholding pipeline for PyTesseract.
    Best for: most real-world conditions (shadows, glare, uneven lighting).
    
    Top-Hat extracts objects (text) brighter than their immediate surroundings,
    effectively erasing dynamic shadows. Adaptive thresholding calculates local
    binarization cutoffs instead of one global threshold, preventing bright glare
    from blowing out the entire plate.
    """
    # Upscale for PyTesseract (needs ~300 DPI equivalent resolution)
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # Clear plate frame borders (5% inward crop)
    cleared = _clear_borders(upscaled, margin_pct=0.05)
    
    # Top-Hat Transform: isolates bright white text, erases background shadows
    # Kernel sized to match typical license plate character proportions (wide, short)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    tophat = cv2.morphologyEx(cleared, cv2.MORPH_TOPHAT, kernel)
    contrast_enhanced = cv2.add(cleared, tophat)
    
    # Adaptive Thresholding: calculates local binarization cutoffs per region
    # instead of one global Otsu threshold — prevents glare blowouts
    binary = cv2.adaptiveThreshold(
        contrast_enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Auto-invert so PyTesseract always gets black text on white background
    binary = _auto_invert(binary)
    
    # Clean minor noise specks
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    
    return _add_white_padding(binary)


def _preprocess_dog_tess(gray):
    """
    DoG illumination normalization + Adaptive Thresholding for PyTesseract.
    Best for: partial shadows, half-lit plates, uneven lighting gradients.
    """
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    cleared = _clear_borders(upscaled, margin_pct=0.05)
    
    # DoG flattens lighting gradients across the plate surface
    dog = _normalize_illumination_dog(cleared)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(dog)
    
    binary = cv2.adaptiveThreshold(
        clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    binary = _auto_invert(binary)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return _add_white_padding(binary)


def _preprocess_aggressive_clahe_tess(gray):
    """
    Aggressive CLAHE (high clipLimit, small tiles) + Adaptive Thresholding.
    Best for: very low contrast, fog, dirty/faded plates, night shots.
    """
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    cleared = _clear_borders(upscaled, margin_pct=0.05)
    
    gamma_corr = _apply_adaptive_gamma(cleared)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4)).apply(gamma_corr)
    
    binary = cv2.adaptiveThreshold(
        clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    binary = _auto_invert(binary)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return _add_white_padding(binary)


# === EASYOCR STRATEGIES ===
# EasyOCR uses a neural network — do NOT binarize. Preserve grayscale gradients
# and anti-aliased character edges. Bilateral filter removes noise while keeping edges.

def _preprocess_standard_easy(gray):
    """
    Standard EasyOCR pipeline: Gamma → Bilateral → CLAHE → 2x Upscale.
    Best for: normal daylight conditions.
    """
    gamma_corr = _apply_adaptive_gamma(gray)
    filtered = cv2.bilateralFilter(gamma_corr, d=11, sigmaColor=17, sigmaSpace=17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(filtered)
    upscaled = cv2.resize(clahe, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return _add_white_padding(upscaled)


def _preprocess_dog_easy(gray):
    """
    DoG illumination normalization → Bilateral → CLAHE → 2x Upscale.
    Best for: partial shadows, uneven lighting.
    """
    dog = _normalize_illumination_dog(gray)
    filtered = cv2.bilateralFilter(dog, d=11, sigmaColor=17, sigmaSpace=17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(filtered)
    upscaled = cv2.resize(clahe, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return _add_white_padding(upscaled)


def _preprocess_aggressive_clahe_easy(gray):
    """
    Aggressive CLAHE (5.0) with smaller tiles → Bilateral → 2x Upscale.
    Best for: very low contrast, night, fog.
    """
    gamma_corr = _apply_adaptive_gamma(gray)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4)).apply(gamma_corr)
    filtered = cv2.bilateralFilter(clahe, d=11, sigmaColor=17, sigmaSpace=17)
    upscaled = cv2.resize(filtered, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return _add_white_padding(upscaled)


# ─── Public Preprocessing (kept for backward compatibility) ───────

def preprocess_for_tesseract(cropped_plate_img):
    """
    PyTesseract preprocessing using Top-Hat Transform + Adaptive Thresholding.
    Strips away real-world lighting and plate frames to produce clean binarized output.
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    gray = cv2.cvtColor(cropped_plate_img, cv2.COLOR_BGR2GRAY) if len(cropped_plate_img.shape) == 3 else cropped_plate_img.copy()
    result = _preprocess_tophat_tess(gray)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def preprocess_for_easyocr(cropped_plate_img):
    """
    EasyOCR preprocessing: Bilateral filter + CLAHE + 2x upscale.
    Preserves natural grayscale gradients that the neural network needs.
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    gray = cv2.cvtColor(cropped_plate_img, cv2.COLOR_BGR2GRAY) if len(cropped_plate_img.shape) == 3 else cropped_plate_img.copy()
    result = _preprocess_standard_easy(gray)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def preprocess_plate_crop(cropped_plate_img, target_width=300):
    return preprocess_for_tesseract(cropped_plate_img)


# ─── OCR Post-Processing & Engine Runners ─────────────────────────

def _postprocess_ocr_text(combined_text, avg_conf):
    """
    Validates the OCR string format and scales confidence values.
    """
    compact = PLATE_CHAR_PATTERN.sub('', combined_text.strip().upper())
    if len(compact) < MIN_PLATE_LENGTH or len(compact) > MAX_PLATE_LENGTH:
        return '', 0.0

    adjusted_conf = max(0.01, avg_conf * min(1.0, len(compact) / 7.0)) if compact else 0.0
    formatted = _fix_plate_format(compact)
    return formatted, adjusted_conf

def _run_pytesseract_raw(processed):
    """
    Executes raw PyTesseract image-to-text detection on a preprocessed image.
    
    Uses --psm 7 (single text line) and a character whitelist to prevent
    punctuation hallucinations. Returns word-level confidence for scoring.
    """
    try:
        # --psm 7: Treat the image as a single horizontal text line (ideal for license plates)
        # --oem 1: Use LSTM neural network OCR engine
        # tessedit_char_whitelist: Restrict output to uppercase letters + digits only
        config = "--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        data_dict = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        
        words_confs = []
        for word, confidence in zip(data_dict.get('text', []), data_dict.get('conf', [])):
            # Filter non-alphanumeric noise characters
            cleaned = PLATE_CHAR_PATTERN.sub('', str(word).strip().upper())
            if cleaned:
                try:
                    val = int(float(confidence))
                except (ValueError, TypeError):
                    val = -1
                words_confs.append((cleaned, 50 if val == -1 else val))
        
        words = [word_confidence_pair[0] for word_confidence_pair in words_confs]
        confidences = [word_confidence_pair[1] for word_confidence_pair in words_confs]
        combined_text = " ".join(words)
        # Scale PyTesseract 0-100 confidence score down to a 0.0 - 1.0 float
        avg_conf = (float(np.mean(confidences)) / 100.0) if confidences else 0.0
        return combined_text, avg_conf
    except Exception as e:
        print(f"[PyTesseract WARN] {e}")
        return '', 0.0

def _run_easyocr_raw(processed):
    """
    Executes raw EasyOCR detection on a preprocessed image.
    
    Restricts recognized characters to uppercase letters and digits via allowlist,
    eliminating punctuation noise/hallucinations.
    """
    try:
        # Allowlist restricts EasyOCR to uppercase letters (A-Z) and digits (0-9)
        allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        # text_threshold=0.3: Minimum character confidence score
        # low_text=0.2: Threshold for grouping nearby letters into a line
        # mag_ratio=1.5: Upscales internal crop size to capture details in smaller text
        results = reader.readtext(processed, allowlist=allowlist, paragraph=False, text_threshold=0.3, low_text=0.2, mag_ratio=1.5)
        if not results:
            return '', 0.0

        results_sorted = sorted(results, key=lambda detection_result: (round(detection_result[0][0][1] / 20), detection_result[0][0][0]))
        combined_text = PLATE_CHAR_PATTERN.sub('', " ".join(detection_result[1].upper().strip() for detection_result in results_sorted))
        confidences = [float(detection_result[2]) for detection_result in results_sorted if len(detection_result) > 2]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return combined_text, avg_conf
    except Exception as e:
        print(f"[EasyOCR WARN] {e}")
        return '', 0.0

def _run_ocr(processed, engine_name):
    """
    Wraps engine-specific raw OCR reads with shared format validation and scoring.
    """
    if engine_name == "PyTesseract":
        combined_text, avg_conf = _run_pytesseract_raw(processed)
    else:
        combined_text, avg_conf = _run_easyocr_raw(processed)
    return _postprocess_ocr_text(combined_text, avg_conf)


def _score_ocr_result(text, conf):
    """
    Scores an OCR result by combining confidence with a format validity bonus.
    Results matching Malaysian plate format get a 1.5x bonus so the fusion
    prefers properly-formatted reads over high-confidence garbage.
    """
    if not text or conf <= 0:
        return 0.0
    score = conf
    # Bonus for matching expected Malaysian plate format
    if MALAYSIAN_PLATE_REGEX.match(text):
        score *= 1.5
    # Small bonus for being in the sweet-spot length range (5-8 chars typical)
    compact_len = len(text.replace(' ', ''))
    if 5 <= compact_len <= 8:
        score *= 1.1
    return score


# ─── Main Entry Point: Multi-Variant OCR with Fusion ─────────────

def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """
    Multi-Variant OCR Dispatcher:
    Generates 3 preprocessed variants of the plate crop, runs OCR on each,
    and picks the best result by score.
    
    PyTesseract variants use Top-Hat Transform + Adaptive Thresholding (not Otsu)
    for proper binarization that handles uneven lighting.
    
    EasyOCR variants preserve grayscale gradients with Bilateral + CLAHE
    since the neural network needs natural anti-aliased edges.
    
    This makes extraction robust across all lighting: daylight, night, partial
    shadow, fog, overexposed, backlighting, etc.
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return '', 0.0, engine_name, None

    # Convert to grayscale once, shared by all variants
    gray = cv2.cvtColor(cropped_plate_img, cv2.COLOR_BGR2GRAY) if len(cropped_plate_img.shape) == 3 else cropped_plate_img.copy()

    # Upscale tiny crops and stretch low-range histograms before any processing
    gray = _upscale_if_small(gray, min_height=40)
    gray = _stretch_histogram(gray)

    # Deskew tilted plates
    gray = _deskew_plate(gray)

    # Build list of preprocessing strategies based on engine
    if engine_name == "PyTesseract":
        strategies = [
            ("tophat", _preprocess_tophat_tess),
            ("dog", _preprocess_dog_tess),
            ("aggressive_clahe", _preprocess_aggressive_clahe_tess),
        ]
    else:
        strategies = [
            ("standard", _preprocess_standard_easy),
            ("dog", _preprocess_dog_easy),
            ("aggressive_clahe", _preprocess_aggressive_clahe_easy),
        ]

    best_text = ''
    best_conf = 0.0
    best_score = 0.0
    best_img = None

    for strategy_name, preprocess_fn in strategies:
        try:
            processed_gray = preprocess_fn(gray)
            if processed_gray is None:
                continue
            processed_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)

            text, conf = _run_ocr(processed_rgb, engine_name)
            score = _score_ocr_result(text, conf)

            if score > best_score:
                best_text = text
                best_conf = conf
                best_score = score
                best_img = processed_rgb
        except Exception as e:
            print(f"[OCR Variant WARN] {strategy_name}: {e}")
            continue

    # If all variants failed, fall back to the standard preprocessed image
    if best_img is None:
        if engine_name == "PyTesseract":
            best_img = preprocess_for_tesseract(cropped_plate_img)
        else:
            best_img = preprocess_for_easyocr(cropped_plate_img)

    return best_text, best_conf, engine_name, best_img
