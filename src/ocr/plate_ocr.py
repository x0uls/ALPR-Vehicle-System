import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import time
import cv2
import numpy as np
import easyocr
import torch
import pytesseract

# Automatically locate Tesseract-OCR executable binary across Linux and Windows default paths
for t_path in ["/usr/bin/tesseract", "/usr/local/bin/tesseract", r"C:\Program Files\Tesseract-OCR\tesseract.exe"]:
    if os.path.exists(t_path):
        pytesseract.pytesseract.tesseract_cmd = t_path
        break

# Initialize EasyOCR reader model instance once (reused across all incoming image requests)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Regular expressions and constants for license plate post-processing
PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9]') # Keeps alphanumeric characters only
MIN_PLATE_LENGTH = 3                           # Minimum valid plate length
MAX_PLATE_LENGTH = 10                          # Maximum valid plate length
_TESS_CONFIG = "--psm 7 --oem 1"               # PSM 7: Treats image as a single text line; OEM 1: Neural net LSTM engine


def _fix_plate_format(text):
    """
    Applies domain-specific Malaysian / standard license plate formatting heuristics:
    1. Removes all non-alphanumeric noise characters.
    2. Corrects inverted readings (e.g., "9393WSX" -> "WSX9393").
    3. Positional character translation:
       - Prefix positions (0..1st digit): Converts lookalike numbers to letters (e.g., '0'->'O', '1'->'I', '5'->'S').
       - Number positions (1st digit..last digit): Converts lookalike letters to numbers (e.g., 'O'->'0', 'I'->'1', 'B'->'8').
    4. Inserts standard space delimiter between letters and numbers.
    """
    compact = PLATE_CHAR_PATTERN.sub('', text.strip().upper()).replace(' ', '')
    if len(compact) < MIN_PLATE_LENGTH:
        return text

    # Handle inverted text reading (e.g., "9393WSX" -> "WSX9393") for 3-4 digits followed by 2-3 letters
    if compact[0].isdigit():
        m = re.match(r'^(\d{3,4})([A-Z]{2,3})$', compact)
        if m:
            digits, letters = m.group(1), m.group(2)
            compact = letters + digits

    # Translation tables for character position corrections
    d2l = str.maketrans('01258', 'OIZSB') # Digits-to-Letters mapping for prefix/suffix sections
    l2d = str.maketrans('OIZSB', '01258') # Letters-to-Digits mapping for numerical plate section

    chars = list(compact)
    if chars[0].isdigit():
        chars[0] = chars[0].translate(d2l)

    # Locate boundary indices of the central numerical segment
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

    # Insert spacing between alphabetical prefix and numerical string
    result = "".join(chars)
    for i in range(1, len(result)):
        if result[i-1].isalpha() and result[i].isdigit():
            return result[:i] + ' ' + result[i:]
    return result


def _add_white_padding(img, pad=15):
    """
    Adds a uniform 15px white border around the image crop.
    Why: OCR engines (especially PyTesseract) misinterpret characters touching image borders as noise/lines.
    Adding white padding provides clean margins around character edges.
    """
    border_color = (255, 255, 255) if len(img.shape) == 3 else 255
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border_color)


def _upscale_and_denoise_first(crop, target_h=140):
    """
    Step 1: Rescaling & Denoising Pipeline.
    - Scale Factor: Upscales small plate crops to a target height of 140px so characters have enough pixel resolution.
    - Interpolation (cv2.INTER_LANCZOS4): Uses Lanczos sinc resampling over an 8x8 neighborhood to prevent pixelation/artifacts.
    - Denoising (cv2.bilateralFilter): Non-linear filter that smooths background sensor noise while preserving sharp character edge boundaries.
      * d=7: Pixel neighborhood diameter for filtering.
      * sigmaColor=75: Filter sigma in color space; smooths minor color fluctuations while respecting strong edge contrast.
      * sigmaSpace=75: Filter sigma in coordinate space; controls spatial smoothing extent.
    """
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop

    # Calculate scale factor to reach 140px height (minimum scale of 3.0x)
    scale = max(3.0, target_h / float(h))
    up = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    try:
        smooth = cv2.bilateralFilter(up, d=7, sigmaColor=75, sigmaSpace=75)
        return smooth
    except Exception:
        return up


def _preprocess_clean_2x(crop):
    """
    Pre-processing pipeline for EasyOCR:
    1. Upscales and denoises crop to 140px target height.
    2. Adds white margin padding for border character isolation.
    """
    highres = _upscale_and_denoise_first(crop, target_h=140)
    return _add_white_padding(highres, pad=15)


def _preprocess_tesseract_single_pass(crop):
    """
    Pre-processing pipeline optimized specifically for PyTesseract line recognition:
    """
    if crop is None or crop.size == 0:
        return crop

    # 1. Upscale and denoise image to high resolution
    highres = _upscale_and_denoise_first(crop, target_h=140)

    # Convert RGB/BGR color image to 8-bit single-channel grayscale
    gray = cv2.cvtColor(highres, cv2.COLOR_BGR2GRAY) if len(highres.shape) == 3 else highres

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # - clipLimit=2.0: Threshold for contrast limiting; prevents oversaturation of noise/glare.
    # - tileGridSize=(8, 8): Divides image into an 8x8 grid of local contextual regions for localized contrast boost.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Mild Sharpening via 2D Convolution Kernel
    # Kernel Matrix: [[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]]
    # Why these values are chosen:
    # - Center weight (+3.0): Strongly amplifies the intensity of the target pixel.
    # - Surrounding weights (-0.5 top/bottom/left/right): Subtracts adjacent pixel values, sharpening edge transitions.
    # - Kernel Sum: 3.0 - (4 * 0.5) = 1.0 (Unit Gain: preserves overall image brightness without darkening or blinding).
    sharpen_k = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_k)

    # 4. Otsu Binarization (Thresholding)
    # - cv2.GaussianBlur(..., (3,3), 0): Applies light Gaussian blur to eliminate high-frequency sharpening artifacts.
    # - cv2.THRESH_BINARY + cv2.THRESH_OTSU: Calculates optimal global threshold value by minimizing intra-class variance.
    blur = cv2.GaussianBlur(sharpened, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Polarity Check (Inversion)
    # PyTesseract expects dark characters on a white background.
    # If the mean pixel intensity < 127 (mostly dark background), bitwise invert to make text dark and background white.
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # Add white margin padding and convert 1-channel grayscale to 3-channel RGB image
    return cv2.cvtColor(_add_white_padding(thresh, pad=15), cv2.COLOR_GRAY2RGB)


def _postprocess_ocr_text(combined_text, avg_conf):
    """
    Cleans raw OCR engine outputs:
    1. Removes non-alphanumeric characters.
    2. Validates character length constraints (3..10 chars).
    3. Applies plate formatting rules.
    4. Calculates length-adjusted confidence score.
    """
    compact = PLATE_CHAR_PATTERN.sub('', combined_text.strip().upper()).replace(' ', '')
    if not (MIN_PLATE_LENGTH <= len(compact) <= MAX_PLATE_LENGTH):
        return '', 0.0
    return _fix_plate_format(compact), max(0.01, avg_conf * min(1.0, len(compact) / 7.0))


def _run_pytesseract_raw(processed):
    """
    Runs PyTesseract OCR extraction on the pre-processed plate crop.
    Returns (raw_text_string, average_confidence_score).
    """
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
        return "".join(words), (float(np.mean(confs)) / 100.0) if confs else 0.0
    except Exception as e:
        print(f"[PyTesseract WARN] {e}")
        return '', 0.0


def _run_easyocr_raw(processed):
    """
    Runs EasyOCR text extraction on the pre-processed plate crop.
    Sorts detected text blocks left-to-right to ensure correct character ordering.
    Returns (raw_text_string, average_confidence_score).
    """
    try:
        results = reader.readtext(processed)
        if not results: return '', 0.0
        sorted_results = sorted(results, key=lambda r: r[0][0][0]) # Sort bounding boxes left-to-right by X-coordinate
        combined_text = "".join([r[1].upper().strip() for r in sorted_results])
        cleaned = PLATE_CHAR_PATTERN.sub('', combined_text)
        confs = [float(r[2]) for r in sorted_results if len(r) > 2]
        return cleaned, float(np.mean(confs)) if confs else 0.0
    except Exception as e:
        print(f"[EasyOCR WARN] {e}")
        return '', 0.0


def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """
    Main entry point for plate OCR recognition.
    Dispatches plate crops to specified engine (EasyOCR or PyTesseract), measures execution latency,
    and returns (plate_text, confidence, engine_name, processed_image_crop, latency_ms).
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return '', 0.0, engine_name, None, 0.0

    t_start = time.perf_counter()
    try:
        if engine_name == "PyTesseract":
            proc = _preprocess_tesseract_single_pass(cropped_plate_img)
            raw_text, conf = _run_pytesseract_raw(proc)
        else:
            proc = _preprocess_clean_2x(cropped_plate_img)
            raw_text, conf = _run_easyocr_raw(proc)

        final_text, final_conf = _postprocess_ocr_text(raw_text, conf)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return final_text, final_conf, engine_name, proc, latency_ms

    except Exception as e:
        print(f"[OCR WARN] {engine_name}: {e}")
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return '', 0.0, engine_name, None, latency_ms
