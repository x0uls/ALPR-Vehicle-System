import os
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

# Initialize EasyOCR reader once at import time (thread-safe for downstream worker pools)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9 ]')
MIN_PLATE_LENGTH = 4
MAX_PLATE_LENGTH = 10

SPECIAL_PLATE_PREFIXES = ["PUTRAJAYA", "RIMAU", "1M4U", "PERODUA", "PROTON"]
MALAYSIAN_PLATE_REGEX = re.compile(
    r'^(' + '|'.join(SPECIAL_PLATE_PREFIXES) + r'|[A-Z]{1,3})\s?\d{1,4}(\s?[A-Z])?$'
)

# Shared Tesseract + EasyOCR character sets
_ALPHANUMERIC_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
_TESS_CONFIG = "--psm 7 --oem 1"


def _fix_plate_format(text):
    """Corrects common OCR letter/digit confusion based on Malaysian license plate formats."""
    compact = text.replace(' ', '').upper()
    if len(compact) < MIN_PLATE_LENGTH:
        return text

    d2l = str.maketrans('01258', 'OIZSB')
    l2d = str.maketrans('OIZSB', '01258')
    extra_l2d = {'G': '6', 'D': '0', 'Q': '0', 'T': '7'}

    # Malaysian plates ALWAYS start with a letter — intercept first-char hallucinations
    if compact[0].isdigit():
        compact = compact[0].translate(d2l) + compact[1:]

    first_dig = next((i for i, c in enumerate(compact) if c.isdigit()), None)
    if first_dig is None:
        return text
    last_dig = max(i for i, c in enumerate(compact) if c.isdigit())

    out = []
    for i, c in enumerate(compact):
        if i < first_dig:
            out.append(c.translate(d2l) if c.isdigit() else c)
        elif i <= last_dig:
            if c.isalpha():
                c = extra_l2d.get(c, c.translate(l2d))
            out.append(c)
        else:
            out.append(c.translate(d2l) if c.isdigit() else c)

    result = ''.join(out)
    # Insert space between letter prefix and digit block
    for i in range(1, len(result)):
        if result[i-1].isalpha() and result[i].isdigit():
            return result[:i] + ' ' + result[i:]
    return result


# ─── Illumination & Preprocessing Helpers ─────────────────────────

def _apply_adaptive_gamma(gray):
    """4-band adaptive gamma correction for extreme lighting normalization."""
    mean_val = np.mean(gray)
    if   mean_val < 50:  gamma = 0.45
    elif mean_val < 90:  gamma = 0.65
    elif mean_val > 210: gamma = 1.8
    elif mean_val > 170: gamma = 1.4
    else: return gray

    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray, table)


def _normalize_illumination_dog(gray):
    """DoG illumination normalization — flattens uneven lighting gradients."""
    dog = cv2.subtract(
        cv2.GaussianBlur(gray, (3, 3), 1),
        cv2.GaussianBlur(gray, (51, 51), 20)
    )
    return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _clear_borders(gray, margin_pct=0.05):
    """Shaves outer 5% to remove physical plate frame that causes OCR hallucinations."""
    h, w = gray.shape[:2]
    my, mx = int(h * margin_pct), int(w * margin_pct)
    if h - 2 * my < 10 or w - 2 * mx < 20:
        return gray
    return gray[my:h - my, mx:w - mx]


def _deskew_plate(gray):
    """Corrects plate rotation up to ±15° using Hough line detection."""
    h, w = gray.shape[:2]
    if h < 10 or w < 20:
        return gray

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=w // 4, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return gray

    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1))
              for (x1, y1, x2, y2), in lines
              if (x2 - x1) != 0 and abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) < 15]

    if not angles:
        return gray
    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return gray

    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=int(np.median(gray)))


def _upscale_if_small(gray, min_height=40):
    """Upscales small plate crops so OCR engines have enough pixel detail."""
    h = gray.shape[0]
    if h < min_height:
        scale = min_height / float(h)
        return cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def _stretch_histogram(gray):
    """Stretches low dynamic range (fog/haze/washed-out) to full 0-255."""
    pmin, pmax = np.percentile(gray, (2, 98))
    if pmax - pmin < 30:
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.clip((gray.astype(np.float32) - pmin) * 255.0 / max(1, pmax - pmin), 0, 255).astype(np.uint8)


def _auto_invert(binary):
    """Inverts binary image if background is dark (ensures black-text-on-white for Tesseract)."""
    return cv2.bitwise_not(binary) if np.median(binary) < 127 else binary


def _add_white_padding(img, pad=15):
    """Adds white border padding for better OCR boundary detection."""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


# Reusable morphological cleanup kernel
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

def _clean_and_pad(binary):
    """Morphological open (denoise) → auto-invert → white padding."""
    binary = _auto_invert(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _MORPH_KERNEL, iterations=1)
    return _add_white_padding(binary)


# ─── Tesseract Strategies (binarized output) ─────────────────────

def _tess_upscale_and_clear(gray):
    """Shared first step for all Tesseract strategies: 2x upscale + border shave."""
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return _clear_borders(upscaled, margin_pct=0.05)


def _adaptive_binarize(img):
    """Gaussian adaptive threshold → auto-invert → denoise → pad."""
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return _clean_and_pad(binary)


def _preprocess_tophat_tess(gray):
    """Top-Hat + Adaptive Threshold. Best for: shadows, glare, uneven lighting."""
    cleared = _tess_upscale_and_clear(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    tophat = cv2.morphologyEx(cleared, cv2.MORPH_TOPHAT, kernel)
    return _adaptive_binarize(cv2.add(cleared, tophat))


def _preprocess_dog_tess(gray):
    """DoG normalization + CLAHE + Adaptive Threshold. Best for: partial shadows."""
    cleared = _tess_upscale_and_clear(gray)
    dog = _normalize_illumination_dog(cleared)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(dog)
    return _adaptive_binarize(clahe)


def _preprocess_aggressive_clahe_tess(gray):
    """Aggressive CLAHE + Gamma + Adaptive Threshold. Best for: night, fog, low contrast."""
    cleared = _tess_upscale_and_clear(gray)
    gamma_corr = _apply_adaptive_gamma(cleared)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4)).apply(gamma_corr)
    return _adaptive_binarize(clahe)


# ─── EasyOCR Strategies (grayscale output, no binarization) ──────

def _easy_filter_and_upscale(gray_input):
    """Shared: bilateral filter → 2x upscale → padding."""
    filtered = cv2.bilateralFilter(gray_input, d=11, sigmaColor=17, sigmaSpace=17)
    upscaled = cv2.resize(filtered, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return _add_white_padding(upscaled)


def _preprocess_standard_easy(gray):
    """Gamma → Bilateral → CLAHE → 2x Upscale. Best for: normal daylight."""
    gamma_corr = _apply_adaptive_gamma(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gamma_corr)
    return _easy_filter_and_upscale(clahe)


def _preprocess_dog_easy(gray):
    """DoG → Bilateral → CLAHE → 2x Upscale. Best for: partial shadows."""
    dog = _normalize_illumination_dog(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(dog)
    return _easy_filter_and_upscale(clahe)


def _preprocess_aggressive_clahe_easy(gray):
    """Aggressive CLAHE → Bilateral → 2x Upscale. Best for: night, fog."""
    gamma_corr = _apply_adaptive_gamma(gray)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4)).apply(gamma_corr)
    return _easy_filter_and_upscale(clahe)


# ─── OCR Engine Runners ──────────────────────────────────────────

def _postprocess_ocr_text(combined_text, avg_conf):
    """Validates OCR string format, applies plate format correction, and scales confidence."""
    compact = PLATE_CHAR_PATTERN.sub('', combined_text.strip().upper())
    if not (MIN_PLATE_LENGTH <= len(compact) <= MAX_PLATE_LENGTH):
        return '', 0.0
    adjusted_conf = max(0.01, avg_conf * min(1.0, len(compact) / 7.0))
    return _fix_plate_format(compact), adjusted_conf


def _run_pytesseract_raw(processed):
    """Runs PyTesseract with PSM 7 (single line) and character whitelist."""
    try:
        data = pytesseract.image_to_data(processed, config=_TESS_CONFIG, output_type=pytesseract.Output.DICT)
        words, confs = [], []
        for word, conf in zip(data.get('text', []), data.get('conf', [])):
            cleaned = PLATE_CHAR_PATTERN.sub('', str(word).strip().upper())
            if cleaned:
                try: val = int(float(conf))
                except (ValueError, TypeError): val = -1
                words.append(cleaned)
                confs.append(50 if val == -1 else val)
        combined = " ".join(words)
        avg_conf = (float(np.mean(confs)) / 100.0) if confs else 0.0
        return combined, avg_conf
    except Exception as e:
        print(f"[PyTesseract WARN] {e}")
        return '', 0.0


def _run_easyocr_raw(processed):
    """Runs EasyOCR with alphanumeric allowlist to prevent punctuation hallucinations."""
    try:
        results = reader.readtext(processed, allowlist=_ALPHANUMERIC_ALLOWLIST, paragraph=False,
                                  text_threshold=0.3, low_text=0.2, mag_ratio=1.5)
        if not results:
            return '', 0.0
        results.sort(key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
        combined = PLATE_CHAR_PATTERN.sub('', " ".join(r[1].upper().strip() for r in results))
        confs = [float(r[2]) for r in results if len(r) > 2]
        return combined, (float(np.mean(confs)) if confs else 0.0)
    except Exception as e:
        print(f"[EasyOCR WARN] {e}")
        return '', 0.0


def _run_ocr(processed, engine_name):
    """Runs the specified OCR engine and post-processes the result."""
    if engine_name == "PyTesseract":
        text, conf = _run_pytesseract_raw(processed)
    else:
        text, conf = _run_easyocr_raw(processed)
    return _postprocess_ocr_text(text, conf)


def _score_ocr_result(text, conf):
    """Scores OCR result: confidence × format bonus × length bonus."""
    if not text or conf <= 0:
        return 0.0
    score = conf
    if MALAYSIAN_PLATE_REGEX.match(text):
        score *= 1.5
    compact_len = len(text.replace(' ', ''))
    if 5 <= compact_len <= 8:
        score *= 1.1
    return score


# ─── Strategy Maps ───────────────────────────────────────────────

_TESS_STRATEGIES = [
    ("tophat", _preprocess_tophat_tess),
    ("dog", _preprocess_dog_tess),
    ("aggressive_clahe", _preprocess_aggressive_clahe_tess),
]

_EASY_STRATEGIES = [
    ("standard", _preprocess_standard_easy),
    ("dog", _preprocess_dog_easy),
    ("aggressive_clahe", _preprocess_aggressive_clahe_easy),
]


# ─── Main Entry Point ───────────────────────────────────────────

def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """
    Multi-Variant OCR: runs 3 preprocessing strategies, OCRs each, picks the best.
    Handles all lighting conditions: daylight, night, shadow, fog, glare, etc.
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return '', 0.0, engine_name, None

    # Convert to grayscale once
    gray = cv2.cvtColor(cropped_plate_img, cv2.COLOR_BGR2GRAY) if len(cropped_plate_img.shape) == 3 else cropped_plate_img.copy()
    gray = _upscale_if_small(_stretch_histogram(gray))
    gray = _deskew_plate(gray)

    strategies = _TESS_STRATEGIES if engine_name == "PyTesseract" else _EASY_STRATEGIES

    best_text, best_conf, best_score, best_img = '', 0.0, 0.0, None

    for name, fn in strategies:
        try:
            processed = fn(gray)
            if processed is None:
                continue
            rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            text, conf = _run_ocr(rgb, engine_name)
            score = _score_ocr_result(text, conf)
            if score > best_score:
                best_text, best_conf, best_score, best_img = text, conf, score, rgb
        except Exception as e:
            print(f"[OCR Variant WARN] {name}: {e}")

    # Fallback: return the first strategy's preprocessed image if all OCR failed
    if best_img is None:
        fallback_fn = strategies[0][1]
        best_img = cv2.cvtColor(fallback_fn(gray), cv2.COLOR_GRAY2RGB)

    return best_text, best_conf, engine_name, best_img
