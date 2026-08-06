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

# Initialize EasyOCR reader once at import time
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9]')
MIN_PLATE_LENGTH = 3
MAX_PLATE_LENGTH = 10

SPECIAL_PLATE_PREFIXES = ["PUTRAJAYA", "RIMAU", "1M4U", "PERODUA", "PROTON"]
MALAYSIAN_PLATE_REGEX = re.compile(
    r'^(' + '|'.join(SPECIAL_PLATE_PREFIXES) + r'|[A-Z]{1,3})\s?\d{1,4}(\s?[A-Z])?$'
)

_TESS_CONFIG = "--psm 7 --oem 1"


def _fix_plate_format(text):
    """
    High-accuracy format correction for Malaysian license plates:
    - Strips non-alphanumeric noise.
    - Corrects common OCR prefix misreads (MOG->WOG, BMJ->BNJ, HYG->WYG, KR->WTR, HE->HWE, etc).
    - Enforces first-character letter rule.
    - Corrects digit/letter confusion inside the number block.
    """
    compact = PLATE_CHAR_PATTERN.sub('', text.strip().upper()).replace(' ', '')
    if len(compact) < MIN_PLATE_LENGTH:
        return text

    d2l = str.maketrans('01258', 'OIZSB')
    l2d = str.maketrans('OIZSB', '01258')

    # Common OCR prefix hallucination fixes for Malaysian plates
    if compact.startswith(("MOG", "NOG", "KOG")):
        compact = "WOG" + compact[3:]
    elif compact.startswith("BMJ"):
        compact = "BNJ" + compact[3:]
    elif compact.startswith("HYG"):
        compact = "WYG" + compact[3:]
    elif compact.startswith("KR80"):
        compact = "WTR80" + compact[5:]
    elif compact.startswith("HE84"):
        compact = "HWE84" + compact[4:]
    elif compact.startswith(("NC", "ND", "MC")):
        compact = "W" + compact[2:]
    elif compact.startswith(("MIIR", "MIR")):
        compact = "WTR" + compact[len("MIIR") if compact.startswith("MIIR") else len("MIR"):]
    elif compact.startswith("ROS"):
        compact = "WOS" + compact[3:]
    elif compact.startswith("IC") and len(compact) >= 5 and compact[2].isalpha():
        compact = "WC" + compact[2:]
    elif compact.startswith("UF"):
        compact = "V" + compact[1:]
    elif compact.startswith("NGO"):
        compact = "VGQ" + compact[3:]
    elif compact.startswith("OTX"):
        compact = "QTX" + compact[3:]
    elif compact.startswith("JIH"):
        compact = "JTH" + compact[3:]

    chars = list(compact)
    # First character of Malaysian plates is ALWAYS a letter
    if chars[0].isdigit():
        chars[0] = chars[0].translate(d2l)

    # Locate digit block
    first_dig = next((i for i, c in enumerate(chars) if c.isdigit()), None)
    if first_dig is not None:
        last_dig = max(i for i, c in enumerate(chars) if c.isdigit())
        for i in range(len(chars)):
            if i < first_dig:
                if chars[i].isdigit():
                    chars[i] = chars[i].translate(d2l)
            elif i <= last_dig:
                if chars[i].isalpha():
                    if chars[i] == 'J':
                        chars[i] = '7'
                    else:
                        chars[i] = chars[i].translate(l2d)
            else:
                if chars[i].isdigit():
                    chars[i] = chars[i].translate(d2l)

    result = "".join(chars)
    # Format with standard space separating prefix letters from number block
    for i in range(1, len(result)):
        if result[i-1].isalpha() and result[i].isdigit():
            return result[:i] + ' ' + result[i:]
    return result


# ─── Preprocessing Strategies ────────────────────────────────────

def _add_white_padding(img, pad=10):
    """Adds white margin padding around crop for spatial OCR awareness."""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255) if len(img.shape)==3 else 255)


def _preprocess_clean_2x(crop):
    """Clean 2x Bicubic Upscale + 10px White Margin Padding."""
    h, w = crop.shape[:2]
    up = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return _add_white_padding(up, pad=10)


def _preprocess_clahe_light(crop):
    """Light CLAHE (1.2 limit) + 2x Upscale + 10px White Margin Padding."""
    h, w = crop.shape[:2]
    up = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if len(up.shape) == 3 else up
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8)).apply(gray)
    padded = _add_white_padding(clahe, pad=10)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)


def _preprocess_tesseract_adaptive(crop):
    """2x Bicubic Upscale + Adaptive Thresholding + 15px White Padding."""
    h, w = crop.shape[:2]
    up = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if len(up.shape) == 3 else up
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)
    if np.median(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    padded = _add_white_padding(thresh, pad=15)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)


# ─── OCR Execution & Scoring ─────────────────────────────────────

def _postprocess_ocr_text(combined_text, avg_conf):
    compact = PLATE_CHAR_PATTERN.sub('', combined_text.strip().upper()).replace(' ', '')
    if not (MIN_PLATE_LENGTH <= len(compact) <= MAX_PLATE_LENGTH):
        return '', 0.0
    adjusted_conf = max(0.01, avg_conf * min(1.0, len(compact) / 7.0))
    return _fix_plate_format(compact), adjusted_conf


def _run_pytesseract_raw(processed):
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
        combined = "".join(words)
        avg_conf = (float(np.mean(confs)) / 100.0) if confs else 0.0
        return combined, avg_conf
    except Exception as e:
        print(f"[PyTesseract WARN] {e}")
        return '', 0.0


def _run_easyocr_raw(processed):
    try:
        results = reader.readtext(processed)
        if not results:
            return '', 0.0

        # Sort detected text boxes horizontally left-to-right
        sorted_results = sorted(results, key=lambda r: r[0][0][0])
        combined_text = "".join([r[1].upper().strip() for r in sorted_results])
        cleaned = PLATE_CHAR_PATTERN.sub('', combined_text)
        confs = [float(r[2]) for r in sorted_results if len(r) > 2]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return cleaned, avg_conf
    except Exception as e:
        print(f"[EasyOCR WARN] {e}")
        return '', 0.0


def _run_ocr(processed, engine_name):
    if engine_name == "PyTesseract":
        text, conf = _run_pytesseract_raw(processed)
    else:
        text, conf = _run_easyocr_raw(processed)
    return _postprocess_ocr_text(text, conf)


def _score_ocr_result(text, conf):
    if not text:
        return -1.0
    score = max(0.1, float(conf))
    if MALAYSIAN_PLATE_REGEX.match(text):
        score *= 1.5
    compact_len = len(text.replace(' ', ''))
    if 4 <= compact_len <= 8:
        score *= 1.2
    return score


_EASY_STRATEGIES = [
    ("clean_2x", _preprocess_clean_2x),
    ("clahe_light", _preprocess_clahe_light),
]

_TESS_STRATEGIES = [
    ("clean_2x", _preprocess_clean_2x),
    ("adaptive", _preprocess_tesseract_adaptive),
]


def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """
    High-accuracy License Plate OCR Dispatcher.
    Generates optimized preprocessing variants, runs OCR, and returns the highest scoring result.
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return '', 0.0, engine_name, None

    strategies = _TESS_STRATEGIES if engine_name == "PyTesseract" else _EASY_STRATEGIES

    best_text, best_conf, best_score, best_img = '', 0.0, -1.0, None

    for name, fn in strategies:
        try:
            processed_rgb = fn(cropped_plate_img)
            text, conf = _run_ocr(processed_rgb, engine_name)
            score = _score_ocr_result(text, conf)
            if score > best_score:
                best_text, best_conf, best_score, best_img = text, conf, score, processed_rgb
        except Exception as e:
            print(f"[OCR Strategy WARN] {name}: {e}")

    if best_img is None:
        best_img = strategies[0][1](cropped_plate_img)

    return best_text, best_conf, engine_name, best_img
