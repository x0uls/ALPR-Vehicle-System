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

# Initialize EasyOCR reader instance ONCE at import time (expensive to create, so reuse it)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), quantize=False)

# Constants & Regex Patterns
# Strips everything except uppercase letters and digits from OCR output
PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9]')
MIN_PLATE_LENGTH = 3
MAX_PLATE_LENGTH = 10
# Tesseract config: PSM 7 = treat image as a single text line, OEM 1 = LSTM neural net engine
_TESS_CONFIG = "--psm 7 --oem 1"


# ── 1. Shared Image Utilities ────────────────────────────────────────────────
def _prepare_base_crop(crop, target_h=140, pad=15):
    """Shared preprocessing for both OCR engines:
    1. Upscales small crops to target height (140px) using Lanczos sinc interpolation
       — this preserves sharp character edges better than bilinear/bicubic.
    2. Applies bilateral filtering to smooth out noise while keeping edges crisp.
    3. Adds a uniform white padding border around the crop — OCR engines perform
       better when text isn't touching the image edges.
    """
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 5 or w < 5:
        return None

    # Scale factor: at least 3x, or whatever gets us to target_h
    scale = max(3.0, target_h / float(h))
    up = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    try:
        # Bilateral filter: d=7 kernel, sigmaColor=75 (how much color diff is tolerated),
        # sigmaSpace=75 (how far pixels influence each other)
        smooth = cv2.bilateralFilter(up, d=7, sigmaColor=75, sigmaSpace=75)
    except Exception:
        smooth = up

    # Add white border padding (helps OCR engines detect text near edges)
    border_color = (255, 255, 255) if len(smooth.shape) == 3 else 255
    return cv2.copyMakeBorder(smooth, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border_color)


# ── 2. EasyOCR Engine Pipeline ───────────────────────────────────────────────
def _process_easyocr(crop):
    """Self-contained EasyOCR pipeline:
    1. Runs shared preprocessing (upscale + denoise + border).
    2. Calls reader.readtext() which returns a list of (bbox, text, confidence) tuples.
    3. Sorts fragments left-to-right by x-coordinate (in case plate text is split).
    4. Joins all fragments into one string and strips non-alphanumeric chars.

    Returns: (cleaned_text, average_confidence, preprocessed_image)
    """
    processed = _prepare_base_crop(crop, target_h=140, pad=15)
    if processed is None:
        return '', 0.0, None

    try:
        # EasyOCR returns list of [bbox_corners, text_string, confidence_float]
        results = reader.readtext(processed)
        if not results:
            return '', 0.0, processed

        # Sort by x-coordinate of top-left corner to get left-to-right reading order
        sorted_results = sorted(results, key=lambda r: r[0][0][0])
        # Concatenate all text fragments into one string
        combined_text = "".join([r[1].upper().strip() for r in sorted_results])
        # Remove any non-alphanumeric characters (dashes, spaces, symbols)
        cleaned = PLATE_CHAR_PATTERN.sub('', combined_text)
        # Average confidence across all detected fragments
        confs = [float(r[2]) for r in sorted_results if len(r) > 2]
        return cleaned, (float(np.mean(confs)) if confs else 0.0), processed
    except Exception as e:
        print(f"[EasyOCR WARN] {e}")
        return '', 0.0, processed


# ── 3. PyTesseract Engine Pipeline ───────────────────────────────────────────
def _process_pytesseract(crop):
    """Self-contained PyTesseract pipeline with heavier preprocessing:
    1. Shared base prep (upscale + denoise + border).
    2. Convert to grayscale.
    3. CLAHE contrast enhancement (adaptive histogram equalization).
    4. Edge sharpening with a custom kernel.
    5. Gaussian blur + Otsu binarization (auto-threshold to black/white).
    6. Polarity check — inverts if background is darker than text.
    7. Runs Tesseract with PSM 7 (single line) and OEM 1 (LSTM engine).

    Returns: (cleaned_text, average_confidence, preprocessed_image)
    """
    base = _prepare_base_crop(crop, target_h=140, pad=15)
    if base is None:
        return '', 0.0, None

    # Convert to grayscale for binarization pipeline
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base

    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    # Boosts local contrast so faded characters become more visible
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpen edges to make character boundaries crisper
    sharpen_k = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_k)

    # Gaussian blur + Otsu binarization: automatically finds the best threshold
    # to separate text (dark) from background (light)
    blur = cv2.GaussianBlur(sharpened, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Polarity check: Tesseract expects dark text on light background
    # If the image is mostly dark (mean < 127), invert it
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # Convert back to 3-channel RGB for Tesseract (it prefers color input)
    processed_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    try:
        # image_to_data returns per-word bounding boxes, text, and confidence
        data = pytesseract.image_to_data(processed_rgb, config=_TESS_CONFIG, output_type=pytesseract.Output.DICT)
        words, confs = [], []
        for word, conf in zip(data.get('text', []), data.get('conf', [])):
            # Clean each word: strip non-alphanumeric, uppercase
            cleaned = PLATE_CHAR_PATTERN.sub('', str(word).strip().upper())
            if cleaned:
                try: val = int(float(conf))
                except (ValueError, TypeError): val = -1
                words.append(cleaned)
                # Tesseract returns -1 for low-confidence words — default those to 50%
                confs.append(50 if val == -1 else val)

        raw_text = "".join(words)
        # Tesseract confidence is 0-100, normalize to 0.0-1.0
        avg_conf = (float(np.mean(confs)) / 100.0) if confs else 0.0
        return raw_text, avg_conf, processed_rgb
    except Exception as e:
        print(f"[PyTesseract WARN] {e}")
        return '', 0.0, processed_rgb


# ── 4. Post-Processing & License Plate Heuristics ────────────────────────────
def _clean_and_format_plate(raw_text, avg_conf):
    """Cleans raw OCR text and applies Malaysian license plate heuristics:
    1. Rejects strings outside length 3-10.
    2. Corrects inverted readings (e.g. "9393WSX" → "WSX9393") — happens when
       OCR reads right-to-left or the plate is mirrored.
    3. Positional character translation: fixes common OCR confusions based on
       whether a position should be a letter or digit (0↔O, 1↔I, 5↔S, 8↔B).
    4. Inserts a space at the letter→digit boundary (e.g. "WD586D" → "WD 586D").
    5. Scales confidence by plate length (shorter plates get penalised).

    Returns: (formatted_plate_text, scaled_confidence)
    """
    # Strip everything non-alphanumeric and uppercase
    compact = PLATE_CHAR_PATTERN.sub('', raw_text.strip().upper()).replace(' ', '')
    if not (MIN_PLATE_LENGTH <= len(compact) <= MAX_PLATE_LENGTH):
        return '', 0.0

    # Inverted reading fix: if plate starts with digits followed by letters,
    # it's likely read backwards — swap them (e.g. "9393WSX" → "WSX9393")
    if compact[0].isdigit():
        m = re.match(r'^(\d{3,4})([A-Z]{2,3})$', compact)
        if m:
            digits, letters = m.group(1), m.group(2)
            compact = letters + digits

    # Translation tables for common OCR character confusions
    # d2l: digit → letter (0→O, 1→I, 2→Z, 5→S, 7→J, 8→B)
    d2l = str.maketrans('012578', 'OIZSJB')
    # l2d: letter → digit (reverse of above)
    l2d = str.maketrans('OIZSJB', '012578')

    chars = list(compact)
    # First character of a Malaysian plate is always a letter
    if chars[0].isdigit():
        chars[0] = chars[0].translate(d2l)

    # Find the digit region and apply positional corrections:
    # - Characters OUTSIDE the digit region should be letters → convert digits to letters
    # - Characters INSIDE the digit region should be digits → convert letters to digits
    first_dig = next((i for i, c in enumerate(chars) if c.isdigit()), None)
    if first_dig is not None:
        last_dig = max(i for i, c in enumerate(chars) if c.isdigit())
        for i in range(len(chars)):
            if i < first_dig or i > last_dig:
                # Position is in the letter zone — fix any digits
                if chars[i].isdigit(): chars[i] = chars[i].translate(d2l)
            else:
                # Position is in the digit zone — fix any letters
                if chars[i].isalpha(): chars[i] = chars[i].translate(l2d)

    # Insert a space at the first letter→digit boundary
    # e.g. "WD586D" → "WD 586D"
    formatted = "".join(chars)
    for i in range(1, len(formatted)):
        if formatted[i-1].isalpha() and formatted[i].isdigit():
            formatted = formatted[:i] + ' ' + formatted[i:]
            break

    # Scale confidence by plate length — shorter plates are less reliable
    # A 7-char plate gets full confidence, shorter plates get proportionally less
    scaled_conf = max(0.01, avg_conf * min(1.0, len(compact) / 7.0))
    return formatted, scaled_conf


# ── 5. Public API ────────────────────────────────────────────────────────────
def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """Main entry point for plate OCR recognition.
    Called by _ocr_worker() in pipeline.py from the thread pools.

    1. Dispatches the crop to either _process_easyocr() or _process_pytesseract().
    2. Runs post-processing via _clean_and_format_plate().
    3. Measures end-to-end latency in milliseconds.

    Returns: (plate_text, confidence, engine_name, processed_image_crop, latency_ms)
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return '', 0.0, engine_name, None, 0.0

    t_start = time.perf_counter()
    try:
        # Dispatch to the appropriate OCR engine
        if engine_name == "PyTesseract":
            raw_text, conf, proc_img = _process_pytesseract(cropped_plate_img)
        else:
            raw_text, conf, proc_img = _process_easyocr(cropped_plate_img)

        # Apply plate-specific post-processing (inversion fix, character correction, formatting)
        final_text, final_conf = _clean_and_format_plate(raw_text, conf)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return final_text, final_conf, engine_name, proc_img, latency_ms

    except Exception as e:
        print(f"[OCR WARN] {engine_name}: {e}")
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return '', 0.0, engine_name, None, latency_ms