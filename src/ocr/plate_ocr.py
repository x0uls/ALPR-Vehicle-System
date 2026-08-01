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


def preprocess_plate_crop(cropped_plate_img, target_width=300):
    """
    Minimal & Clean Plate Preprocessing Pipeline:
    1. Resize while preserving aspect ratio (300px target width).
    2. Convert to Grayscale.
    3. CLAHE Contrast Boost.
    4. Otsu Binarization (optimal global text threshold).
    5. Black text on white background normalization.
    6. Morphological Opening (remove small black noise specks).
    7. White margin padding (15px border).
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    resized = _resize_keep_aspect(cropped_plate_img, target_width, "width")
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    
    # Otsu Binarization: automatically computes optimal threshold for character separation
    _, binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Normalize to black text on white background
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    
    # Morphological Opening (2x2 kernel) to remove small noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 15px white margin padding
    final_img = cv2.copyMakeBorder(binary, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)


def preprocess_raw_bgr(cropped_plate_img, target_height=140):
    """
    High-resolution clean BGR crop with 15px border padding.
    Preserves natural color, font edges, and subtle gradients without harsh binarization.
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None
    h, w = cropped_plate_img.shape[:2]
    aspect = w / h
    target_width = int(target_height * aspect)
    resized = cv2.resize(cropped_plate_img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    if len(resized.shape) == 3:
        padded = cv2.copyMakeBorder(resized, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return padded
    else:
        padded = cv2.copyMakeBorder(resized, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
        return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def preprocess_for_easyocr(cropped_plate_img):
    return preprocess_plate_crop(cropped_plate_img, 300)


def preprocess_for_tesseract(cropped_plate_img, threshold_method="otsu"):
    return preprocess_plate_crop(cropped_plate_img, 300)


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
    Minimal Dual-Pass OCR Engine:
    Pass 1: Clean high-res BGR crop (natural font edges).
    Pass 2: Minimal Otsu-binarized crop (CLAHE + Otsu + 15px border padding).
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return '', 0.0, engine_name, None

    candidates = []
    prep_img = preprocess_plate_crop(cropped_plate_img)

    # Pass 1: Raw high-resolution BGR crop
    raw_img = preprocess_raw_bgr(cropped_plate_img)
    if raw_img is not None:
        text, conf = _run_ocr(raw_img, engine_name)
        if text and conf >= 0.60:
            return text, conf, engine_name, prep_img if prep_img is not None else raw_img
        if text:
            candidates.append((text, conf, prep_img if prep_img is not None else raw_img))

    # Pass 2: Minimal Otsu-binarized crop
    if prep_img is not None:
        text, conf = _run_ocr(prep_img, engine_name)
        if text and conf >= 0.60:
            return text, conf, engine_name, prep_img
        if text:
            candidates.append((text, conf, prep_img))

    if candidates:
        best = max(candidates, key=lambda x: x[1])
        return best[0], best[1], engine_name, best[2]

    return '', 0.0, engine_name, prep_img if prep_img is not None else raw_img
