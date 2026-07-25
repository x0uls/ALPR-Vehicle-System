import os
# Limit OpenMP threads to 1 to prevent CPU thrashing/bottlenecks during inference
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import cv2
import numpy as np
import easyocr
import torch
from deskew import determine_skew
from skimage.transform import rotate
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


def auto_deskew(img):
    """Detects rotational skew in the plate and straightens it."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    angle = determine_skew(gray)
    
    if angle is None or abs(angle) < 1.0:
        return img
        
    # resize=True: expands canvas to prevent clipping text corners; mode='edge': pads border pixels
    rotated = rotate(img, angle, resize=True, mode='edge')
    return (rotated * 255).astype(np.uint8)


def preprocess_for_easyocr(cropped_plate_img):
    """Preprocesses raw plate image optimized for EasyOCR."""
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    resized = _resize_keep_aspect(cropped_plate_img, TARGET_WIDTH, "width")
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    
    # clipLimit=2.0: max contrast boost cap; tileGridSize=(8,8): local grid region size
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    processed = auto_deskew(clahe)
    
    # top/bottom/left/right=15px: white margin padding around plate text
    final_img = cv2.copyMakeBorder(processed, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)


def _crop_plate_text_contours(binary):
    """Isolates valid character contours and crops the text region, excluding outer plate frames."""
    inv_binary = ~binary
    # top/bottom/left/right=10px: temporary black border for closed contour detection
    padded_inv = cv2.copyMakeBorder(inv_binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    padded_height, padded_width = padded_inv.shape[:2]
    contours, _ = cv2.findContours(padded_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [
        c for c in contours 
        if 2 <= cv2.boundingRect(c)[2] < 0.85 * padded_width and 8 <= cv2.boundingRect(c)[3] < 0.85 * padded_height
    ]
    if not valid_contours:
        return binary

    box_x, box_y, box_w, box_h = cv2.boundingRect(np.vstack(valid_contours))
    orig_x1, orig_y1 = max(0, box_x - 10), max(0, box_y - 10)
    orig_x2, orig_y2 = min(binary.shape[1], box_x + box_w - 10), min(binary.shape[0], box_y + box_h - 10)
    
    if orig_x2 > orig_x1 and orig_y2 > orig_y1:
        return binary[orig_y1:orig_y2, orig_x1:orig_x2]
    return binary


def preprocess_for_tesseract(cropped_plate_img, threshold_method="adaptive"):
    """Preprocesses raw plate image optimized for PyTesseract."""
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    resized = _resize_keep_aspect(cropped_plate_img, 150, "height")
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

    deskewed_gray = auto_deskew(gray)
    # d=9: pixel neighborhood diameter; sigmaColor=75 & sigmaSpace=75: intensity & spatial smoothing thresholds
    filtered = cv2.bilateralFilter(deskewed_gray, 9, 75, 75)

    _, test_bin = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(test_bin) < 127:
        filtered = ~filtered

    if threshold_method == "otsu":
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # maxValue=255; 25: local block size (25x25); 2: mean offset constant C
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)

    # (2,2): 2x2 rectangular kernel matrix for speckle removal
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    binary = _crop_plate_text_contours(binary)

    # top/bottom/left/right=20px: white margin padding; (3,3): 3x3 Gaussian kernel matrix for edge softening
    padded_final = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return cv2.GaussianBlur(padded_final, (3, 3), 0)

def _postprocess_ocr_text(combined_text, avg_conf):
    """
    Validates the OCR string format and scales confidence values.
    
    Normalizes spelling, applies plate rules, and discards reads that violate the regex patterns.
    """
    compact = ''.join(combined_text.split())
    # Reject strings that are too short (under 4) or too long (over 10)
    if len(compact) < MIN_PLATE_LENGTH or len(compact) > MAX_PLATE_LENGTH:
        return '', 0.0

    # Scale confidence. If the plate is short, reduce confidence since short reads have a higher chance of error.
    # len(compact)/7.0 applies a linear penalty if length is under 7 characters.
    adjusted_conf = max(0.01, avg_conf * min(1.0, len(compact) / 7.0)) if combined_text.strip() else 0.0
    text = _fix_plate_format(combined_text)

    # Discard OCR readings that fail the Malaysian plate regex pattern
    if not MALAYSIAN_PLATE_REGEX.match(text.replace(' ', '')):
        return '', 0.0

    # Apply a 15% confidence boost (capped at 1.0) because passing the regex is a strong indicator of validity.
    return text, min(1.0, adjusted_conf * 1.15)

def _run_pytesseract_raw(processed):
    """
    Executes raw PyTesseract image-to-text detection on a preprocessed image.
    
    Passes --psm 7 (treat image as single text line) and --oem 1 (LSTM neural network engine).
    """
    try:
        # --psm 7: Treat the image as a single horizontal text line (ideal for license plates)
        # --oem 1: Use LSTM neural network OCR engine
        config = f"--psm 7 --oem 1"
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
    Executes raw EasyOCR detection on a preprocessed image using optimize branch's allowlist and mag_ratio.
    
    Restricts recognized characters to uppercase letters and digits, eliminating punctuation noise.
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

def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """
    Core entry point to extract license plate text and confidences from a plate crop.
    Restored from optimize branch for highest recognition accuracy.
    """
    if engine_name != "PyTesseract":
        processed = preprocess_for_easyocr(cropped_plate_img)
        if processed is None:
            return '', 0.0, engine_name, None
        text, conf = _run_ocr(processed, engine_name)
        return text, conf, engine_name, processed

    candidates = []
    for thresh_method in ("adaptive", "otsu"):
        processed = preprocess_for_tesseract(cropped_plate_img, thresh_method)
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
