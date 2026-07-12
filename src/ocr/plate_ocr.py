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

MALAYSIAN_PLATE_REGEX = re.compile(
    r'^(PUTRAJAYA|RIMAU|1M4U|PERODUA|PROTON|[A-Z]{1,3})\s?\d{1,4}(\s?[A-Z])?$'
)

def _fix_plate_format(text):
    """Fix common OCR misreads based on Malaysian plate format."""
    compact = text.replace(' ', '')
    if len(compact) < MIN_PLATE_LENGTH: return text

    digit_to_letter = str.maketrans('01258', 'OIZSB')
    letter_to_digit = str.maketrans('OIZSB', '01258')
    extra_to_digit = {'G': '6', 'D': '0', 'Q': '0', 'T': '7'}

    first_digit_pos = next((i for i, ch in enumerate(compact) if ch.isdigit()), None)
    if first_digit_pos is None: return text

    last_digit_pos = first_digit_pos
    for i in range(first_digit_pos, len(compact)):
        if compact[i].isdigit(): last_digit_pos = i

    result = []
    for i, ch in enumerate(compact):
        if i < first_digit_pos or i > last_digit_pos:
            result.append(ch.translate(digit_to_letter) if ch.isdigit() else ch)
        else:
            if ch.isalpha():
                result.append(extra_to_digit.get(ch, ch.translate(letter_to_digit)))
            else:
                result.append(ch)

    return ''.join(result)

def auto_deskew(img):
    """Replaces manual contour and matrix rotation math with Hough transforms."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    angle = determine_skew(gray)
    
    if angle is None or abs(angle) < 1.0:
        return img
        
    rotated = rotate(img, angle, resize=True, mode='edge')
    return (rotated * 255).astype(np.uint8)

def preprocess_for_easyocr(cropped_plate_img):
    """Preprocessing optimized for EasyOCR: Grayscale, CLAHE, deskew."""
    if cropped_plate_img is None or cropped_plate_img.size == 0: return None

    # Replaced manual ratio math with imutils
    resized = imutils.resize(cropped_plate_img, width=TARGET_WIDTH)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    processed = auto_deskew(clahe)
    
    final_img = cv2.copyMakeBorder(processed, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

def preprocess_for_tesseract(cropped_plate_img, threshold_method="adaptive"):
    """Clean-room preprocessing pipeline for PyTesseract."""
    if cropped_plate_img is None or cropped_plate_img.size == 0: return None

    gray = cv2.cvtColor(cropped_plate_img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    if threshold_method == "otsu":
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)

    # Automatically wipe edge noise
    cleared = clear_border(cv2.bitwise_not(binary)).astype(np.uint8)
    binary = cv2.bitwise_not(cleared * 255)

    binary = auto_deskew(binary)

    # Polarity Fix (ensure black text on white background)
    if (binary.size - cv2.countNonZero(binary)) > binary.size * 0.5:
        binary = cv2.bitwise_not(binary)

    # Upscale logic via imutils
    if binary.shape[0] < 100:
        binary = imutils.resize(binary, width=int(binary.shape[1] * 2.0))
        
    return cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def _run_ocr(processed, engine_name):
    """Runs the selected OCR engine and applies spatial sorting and text cleaning."""
    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    if engine_name == "PyTesseract":
        try:
            config = f"-c tessedit_char_whitelist={allowlist} --psm 7 --oem 1"
            d = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
            
            words, confidences = [], []
            for i in range(len(d.get('text', []))):
                cleaned_word = PLATE_CHAR_PATTERN.sub('', str(d['text'][i]).strip().upper())
                if not cleaned_word: continue
                    
                val = int(float(d['conf'][i])) if str(d['conf'][i]).replace('.','',1).isdigit() else -1
                if val == -1: val = 50 
                    
                words.append(cleaned_word)
                confidences.append(val)
            
            combined_text = " ".join(words)
            avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
            print(f"[OCR DEBUG] PyTesseract raw='{' '.join(d.get('text', []))}', cleaned='{combined_text}', conf={avg_conf:.3f}")
        except Exception as e:
            print(f"[OCR] PyTesseract execution failed: {str(e)}")
            return '', 0.0
        
    else: # EasyOCR Flow
        results = reader.readtext(processed, allowlist=allowlist, paragraph=False, text_threshold=0.5, low_text=0.3, mag_ratio=1.0)
        if not results: return '', 0.0

        results_sorted = sorted(results, key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
        confidences = [float(r[2]) for r in results_sorted if len(r) > 2]
        
        combined_text = PLATE_CHAR_PATTERN.sub('', " ".join([r[1].upper().strip() for r in results_sorted]))
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

    compact = ' '.join(combined_text.split()).replace(' ', '')
    if len(compact) < MIN_PLATE_LENGTH or len(compact) > MAX_PLATE_LENGTH: return '', 0.0

    adjusted_conf = max(0.01, avg_conf * min(1.0, len(compact) / 7.0)) if combined_text.strip() else 0.0
    text = _fix_plate_format(combined_text)

    # Format validation: boost if valid, penalize if structural garbage
    if MALAYSIAN_PLATE_REGEX.match(text.replace(' ', '')):
        adjusted_conf = min(1.0, adjusted_conf * 1.15)
    else:
        adjusted_conf *= 0.5

    return text, adjusted_conf

def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """Lazy Multi-Variant OCR execution wrapper."""
    if engine_name != "PyTesseract":
        processed = preprocess_for_easyocr(cropped_plate_img)
        if processed is None: return '', 0.0, engine_name, None
        text, conf = _run_ocr(processed, engine_name)
        return text, conf, engine_name, processed

    passes = [
        (preprocess_for_easyocr, None),
        (preprocess_for_tesseract, "otsu"),
        (preprocess_for_tesseract, "adaptive")
    ]
    
    candidates = []
    for preprocess_fn, thresh in passes:
        processed = preprocess_fn(cropped_plate_img) if thresh is None else preprocess_fn(cropped_plate_img, thresh)
        if processed is not None:
            text, conf = _run_ocr(processed, engine_name)
            if text and conf >= 0.50: return text, conf, engine_name, processed
            candidates.append((text, conf, processed))

    valid = [c for c in candidates if c[0].strip()]
    if valid:
        best = max(valid, key=lambda x: x[1])
        return best[0], best[1], engine_name, best[2]
        
    return '', 0.0, engine_name, None
