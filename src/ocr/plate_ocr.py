import os
# Limits OpenMP internal multi-threading to 1 thread to avoid thread oversubscription
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import time
import cv2
import numpy as np
import easyocr
import torch
import pytesseract

# Automatically locate Tesseract-OCR binary across Linux and Windows default paths
for t_path in ["/usr/bin/tesseract", "/usr/local/bin/tesseract", r"C:\Program Files\Tesseract-OCR\tesseract.exe"]:
    if os.path.exists(t_path):
        pytesseract.pytesseract.tesseract_cmd = t_path
        break

# Initialize EasyOCR reader instance ONCE at import time
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), quantize=False)

# Constants & Regex Patterns
PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9]')
MIN_PLATE_LENGTH = 3
MAX_PLATE_LENGTH = 10
_TESS_CONFIG = "--psm 7 --oem 1"


# ── 1. Shared Image Utilities ────────────────────────────────────────────────
def _prepare_base_crop(crop, target_h=140, pad=15):
    """
    Shared base image preparation:
    1. Upscales small crops to target height (140px) using Lanczos sinc interpolation.
    2. Applies bilateral filtering to smooth out noise without blurring character edges.
    3. Adds a uniform white padding border around the crop.
    """
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 5 or w < 5:
        return None

    scale = max(3.0, target_h / float(h))
    up = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    try:
        smooth = cv2.bilateralFilter(up, d=7, sigmaColor=75, sigmaSpace=75)
    except Exception:
        smooth = up

    border_color = (255, 255, 255) if len(smooth.shape) == 3 else 255
    return cv2.copyMakeBorder(smooth, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border_color)


# ── 2. EasyOCR Engine Pipeline ───────────────────────────────────────────────
def _process_easyocr(crop):
    """
    Self-contained EasyOCR pipeline:
    1. Preprocesses crop (Lanczos upscaling + bilateral denoising + white border).
    2. Executes EasyOCR extraction and sorts fragments left-to-right by x-coordinate.
    Returns: (raw_text_string, avg_confidence, preprocessed_image)
    """
    processed = _prepare_base_crop(crop, target_h=140, pad=15)
    if processed is None:
        return '', 0.0, None

    try:
        results = reader.readtext(processed)
        if not results:
            return '', 0.0, processed

        sorted_results = sorted(results, key=lambda r: r[0][0][0])
        combined_text = "".join([r[1].upper().strip() for r in sorted_results])
        cleaned = PLATE_CHAR_PATTERN.sub('', combined_text)
        confs = [float(r[2]) for r in sorted_results if len(r) > 2]
        return cleaned, (float(np.mean(confs)) if confs else 0.0), processed
    except Exception as e:
        print(f"[EasyOCR WARN] {e}")
        return '', 0.0, processed


# ── 3. PyTesseract Engine Pipeline ───────────────────────────────────────────
def _process_pytesseract(crop):
    """
    Self-contained PyTesseract pipeline:
    1. Preprocesses crop: Grayscale -> CLAHE contrast boost -> Sharpening -> Otsu Binarization -> Polarity Check.
    2. Executes Tesseract OCR with PSM 7 (single line) & OEM 1 (LSTM).
    Returns: (raw_text_string, avg_confidence, preprocessed_image)
    """
    base = _prepare_base_crop(crop, target_h=140, pad=15)
    if base is None:
        return '', 0.0, None

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base

    # Local contrast boost (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Edge sharpening
    sharpen_k = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_k)

    # Otsu binarization
    blur = cv2.GaussianBlur(sharpened, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Polarity check (ensure dark characters on light background)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    processed_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    try:
        data = pytesseract.image_to_data(processed_rgb, config=_TESS_CONFIG, output_type=pytesseract.Output.DICT)
        words, confs = [], []
        for word, conf in zip(data.get('text', []), data.get('conf', [])):
            cleaned = PLATE_CHAR_PATTERN.sub('', str(word).strip().upper())
            if cleaned:
                try: val = int(float(conf))
                except (ValueError, TypeError): val = -1
                words.append(cleaned)
                confs.append(50 if val == -1 else val)

        raw_text = "".join(words)
        avg_conf = (float(np.mean(confs)) / 100.0) if confs else 0.0
        return raw_text, avg_conf, processed_rgb
    except Exception as e:
        print(f"[PyTesseract WARN] {e}")
        return '', 0.0, processed_rgb


# ── 4. Post-Processing & License Plate Heuristics ────────────────────────────
def _clean_and_format_plate(raw_text, avg_conf):
    """
    Cleans raw OCR text and applies license plate structure heuristics:
    1. Rejects strings outside length 3-10.
    2. Corrects inverted readings (e.g. "9393WSX" -> "WSX9393").
    3. Positional character translation (0/O, 1/I, 5/S, 8/B).
    4. Formats prefix/number gap (e.g. "WD586D" -> "WD 586D").
    5. Returns (final_text, length_scaled_confidence).
    """
    compact = PLATE_CHAR_PATTERN.sub('', raw_text.strip().upper()).replace(' ', '')
    if not (MIN_PLATE_LENGTH <= len(compact) <= MAX_PLATE_LENGTH):
        return '', 0.0

    # Inverted reading check (numbers before letters)
    if compact[0].isdigit():
        m = re.match(r'^(\d{3,4})([A-Z]{2,3})$', compact)
        if m:
            digits, letters = m.group(1), m.group(2)
            compact = letters + digits

    d2l = str.maketrans('01258', 'OIZSB')
    l2d = str.maketrans('OIZSB', '01258')

    chars = list(compact)
    if chars[0].isdigit():
        chars[0] = chars[0].translate(d2l)

    first_dig = next((i for i, c in enumerate(chars) if c.isdigit()), None)
    if first_dig is not None:
        last_dig = max(i for i, c in enumerate(chars) if c.isdigit())
        for i in range(len(chars)):
            if i < first_dig:
                if chars[i].isdigit(): chars[i] = chars[i].translate(d2l)
            elif i <= last_dig:
                if chars[i].isalpha(): chars[i] = '7' if chars[i] == 'J' else chars[i].translate(l2d)
            else:
                if chars[i].isdigit(): chars[i] = chars[i].translate(d2l)

    formatted = "".join(chars)
    for i in range(1, len(formatted)):
        if formatted[i-1].isalpha() and formatted[i].isdigit():
            formatted = formatted[:i] + ' ' + formatted[i:]
            break

    scaled_conf = max(0.01, avg_conf * min(1.0, len(compact) / 7.0))
    return formatted, scaled_conf


# ── 5. Public API ────────────────────────────────────────────────────────────
def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """
    Main entry point for plate OCR recognition.
    Dispatches crop to EasyOCR or PyTesseract, runs post-processing, and measures latency.

    Returns: (plate_text, confidence, engine_name, processed_image_crop, latency_ms)
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return '', 0.0, engine_name, None, 0.0

    t_start = time.perf_counter()
    try:
        if engine_name == "PyTesseract":
            raw_text, conf, proc_img = _process_pytesseract(cropped_plate_img)
        else:
            raw_text, conf, proc_img = _process_easyocr(cropped_plate_img)

        final_text, final_conf = _clean_and_format_plate(raw_text, conf)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return final_text, final_conf, engine_name, proc_img, latency_ms

    except Exception as e:
        print(f"[OCR WARN] {engine_name}: {e}")
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return '', 0.0, engine_name, None, latency_ms