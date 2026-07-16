import os
# Limit OpenMP threads to 1 to prevent CPU thrashing/bottlenecks during inference
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import cv2
import numpy as np
import easyocr
import torch
from deskew import determine_skew
import pytesseract

# Lazy loader variable to store the EasyOCR model reference
_reader = None

def get_easyocr_reader():
    """
    Initializes and returns the EasyOCR Reader instance on first use (lazy loading).
    
    This avoids consuming VRAM or system memory if the user chooses PyTesseract instead.
    """
    global _reader
    if _reader is None:
        # Load English model. GPU is used automatically if PyTorch detects a CUDA-compatible GPU.
        _reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return _reader


# Regex to strip non-alphanumeric characters, leaving only clean letters, numbers, and spaces
PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9 ]')

# Minimum length for a valid plate (e.g., prefix + numbers like "WND 1")
MIN_PLATE_LENGTH = 4

# Maximum length for a typical license plate
MAX_PLATE_LENGTH = 10

# Scaling target width for EasyOCR. 300 pixels is the sweet spot where characters are
# high-resolution enough to read without blowing up computational/processing time.
TARGET_WIDTH = 300

# Common Malaysian special prefixes. Since these are whole words instead of 1-3 letters,
# we need to recognize them to check their layout structure.
SPECIAL_PLATE_PREFIXES = ["PUTRAJAYA", "RIMAU", "1M4U", "PERODUA", "PROTON"]

# Regular expression checking for Malaysian formats:
# (Special word OR 1-3 letters) + optional space + 1-4 digits + optional single trailing letter
MALAYSIAN_PLATE_REGEX = re.compile(
    r'^(' + '|'.join(SPECIAL_PLATE_PREFIXES) + r'|[A-Z]{1,3})\s?\d{1,4}(\s?[A-Z])?$'
)

def _resize_keep_aspect(img, target, by="width"):
    """
    Resizes an image while preserving its original aspect ratio (width-to-height proportion).
    
    Prevents characters from stretching or compressing, which would confuse the OCR models.
    """
    h, w = img.shape[:2]
    aspect = w / h
    if by == "width":
        return cv2.resize(img, (target, int(target / aspect)))
    return cv2.resize(img, (int(target * aspect), target))

def _rotate_image(image, angle):
    """
    Rotates an image around its center by a specific angle.
    
    Calculates the new bounding box dimensions so character edges aren't cropped/cut off.
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Get the 2D rotation affine matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    # Calculate cosine and sine of the angle to adjust new width and height
    abs_cos_angle = np.abs(rotation_matrix[0, 0])
    abs_sin_angle = np.abs(rotation_matrix[0, 1])
    new_width = int((h * abs_sin_angle) + (w * abs_cos_angle))
    new_height = int((h * abs_cos_angle) + (w * abs_sin_angle))
    
    # Shift the center in the translation columns of the matrix to avoid clipping
    rotation_matrix[0, 2] += (new_width / 2) - center_x
    rotation_matrix[1, 2] += (new_height / 2) - center_y
    
    # Warp the image. BORDER_REPLICATE fills empty corners by copying edge pixels
    return cv2.warpAffine(image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _fix_plate_format(text):
    """
    Corrects common OCR letter/digit confusion based on Malaysian license plate formats.
    
    For example: '0' instead of 'O' in a letters segment, or 'S' instead of '5' in a digits segment.
    It evaluates multiple layouts (letter/digit splits) and picks the one needing the fewest fixes.
    """
    compact = text.replace(' ', '').upper()
    if len(compact) < MIN_PLATE_LENGTH:
        return text

    # Translation maps to fix OCR reading errors based on position expectations
    digit_to_letter = str.maketrans('01258', 'OIZSB')
    letter_to_digit = str.maketrans('OIZSBGDTQ', '012586007')

    # Default to scanning 1, 2, or 3-letter prefix lengths unless we hit a special word prefix
    prefix_lengths = [1, 2, 3]
    for special_prefix in SPECIAL_PLATE_PREFIXES:
        if compact.startswith(special_prefix):
            prefix_lengths = [len(special_prefix)]
            break

    best_corrected = None
    best_changes = 999  # Start with an artificially high count of changes

    # Partition the compact text into prefix (letters), middle (digits), and suffix (optional letter)
    for prefix_length in prefix_lengths:
        for has_suffix in [False, True]:
            if has_suffix:
                if len(compact) <= prefix_length + 1:
                    continue
                prefix = compact[:prefix_length]
                middle_digits = compact[prefix_length:-1]
                suffix = compact[-1]
            else:
                if len(compact) <= prefix_length:
                    continue
                prefix = compact[:prefix_length]
                middle_digits = compact[prefix_length:]
                suffix = ""

            # Malaysian plates always contain between 1 and 4 middle digits
            if not (1 <= len(middle_digits) <= 4):
                continue

            # Apply translations based on expected format
            prefix_corrected = prefix if prefix in SPECIAL_PLATE_PREFIXES else prefix.translate(digit_to_letter)
            middle_digits_corrected = middle_digits.translate(letter_to_digit)
            suffix_corrected = suffix.translate(digit_to_letter) if suffix else ""

            # Check if our corrections result in a valid format layout
            is_valid_prefix = (prefix_corrected in SPECIAL_PLATE_PREFIXES) or (re.match(r'^[A-Z]{1,3}$', prefix_corrected) is not None)
            is_valid_middle = (re.match(r'^\d{1,4}$', middle_digits_corrected) is not None)
            is_valid_suffix = (not suffix_corrected) or (re.match(r'^[A-Z]$', suffix_corrected) is not None)

            if is_valid_prefix and is_valid_middle and is_valid_suffix:
                # Count character mismatches to measure how many corrections were made
                changes = sum(char1 != char2 for char1, char2 in zip(compact, prefix_corrected + middle_digits_corrected + suffix_corrected))
                # Choose the candidate that matches the format rules with the minimum adjustments
                if changes < best_changes:
                    best_changes = changes
                    best_corrected = prefix_corrected + " " + middle_digits_corrected + suffix_corrected

    if best_corrected:
        return best_corrected
    return text

def auto_deskew(img):
    """
    Detects any rotational skew in the plate and straightens it.
    
    Corrects horizontal tilt so text segments align horizontally, enhancing OCR accuracy.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    angle = determine_skew(gray)
    
    # Ignore tiny skew angles (< 1.0 degree) to avoid processing overhead for negligible gains
    if angle is None or abs(angle) < 1.0:
        return img
        
    return _rotate_image(img, angle)

def preprocess_for_easyocr(cropped_plate_img):
    """
    Preprocesses the raw plate image specifically optimized for EasyOCR.
    
    Applies resizing, grayscale, CLAHE (contrast enhancement), and deskew.
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    # Resize to standard width so character thickness matches EasyOCR model training patterns
    resized = _resize_keep_aspect(cropped_plate_img, TARGET_WIDTH, "width")
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    # Improves local contrast in dark/bright spots (shadows/highlights) without blowing out noise.
    # clipLimit=2.0 limits maximum contrast boost, tileGridSize=(8, 8) divides image into local 8x8 blocks.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    processed = auto_deskew(clahe)
    
    # Add a 15-pixel white border so character edges don't touch the image boundaries
    final_img = cv2.copyMakeBorder(processed, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

def preprocess_for_tesseract(cropped_plate_img, threshold_method="adaptive"):
    """
    Preprocesses raw plate image specifically optimized for PyTesseract.
    
    Applies high-DPI scaling, deskewing, noise filtering, polarity inversion, binarization, 
    plate frame contour cropping, and generous white margins.
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return None

    # Step 1: Scale target height to 150px. Tesseract works best at high resolutions (~300 DPI),
    # so upscaling smaller plate crops is necessary for text character detection.
    resized = _resize_keep_aspect(cropped_plate_img, 150, "height")
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

    # Step 2: Auto deskew tilted text
    deskewed_gray = auto_deskew(gray)

    # Step 3: Bilateral Filter. Smoothes out textures while keeping text boundaries sharp.
    # Diameter=9, color sigma=75, space sigma=75 are standard to preserve sharp text outlines.
    filtered = cv2.bilateralFilter(deskewed_gray, 9, 75, 75)

    # Step 4: Detect polarity. If average pixel brightness is dark (< 127), we invert the image.
    # Tesseract expects dark text on a light background; this guarantees consistent polarity.
    _, test_bin = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(test_bin) < 127:
        filtered = ~filtered

    # Step 5: Binarization (convert image to pure black and white)
    if threshold_method == "otsu":
        # Otsu thresholding automatically calculates the optimal threshold value based on histogram bimodal clusters
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Adaptive Gaussian thresholding calculates local thresholds for blocks of size 25x25 (must be odd).
        # Subtracting constant C=2 reduces threshold sensitivity to local salt-and-pepper noise.
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 2
        )

    # Step 5b: Apply Morphological Opening (dilation followed by erosion) using a 2x2 rectangular kernel.
    # Removes tiny speckles and separates characters that may be slightly touching.
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # Step 6: Contour Border Removal & Cropping.
    # Isolates characters by identifying bounded black areas, stripping outer borders of the plate frame.
    inv_binary = ~binary
    # Add a temporary 10px black border to ensure outer borders form closed contours
    padded_inv = cv2.copyMakeBorder(inv_binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    padded_height, padded_width = padded_inv.shape[:2]

    contours, _ = cv2.findContours(padded_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for contour in contours:
        _, _, box_width, box_height = cv2.boundingRect(contour)
        # Exclude tiny noise dots (width < 2 or height < 8) and giant borders (>= 85% of total width/height)
        if 2 <= box_width < 0.85 * padded_width and 8 <= box_height < 0.85 * padded_height:
            valid_contours.append(contour)

    if valid_contours:
        # Stack coordinate points of all valid text contours to extract the unified bounding box around the text
        box_x, box_y, box_width, box_height = cv2.boundingRect(np.vstack(valid_contours))
        margin = 4  # 4px breathing room around bounding box
        x1, y1 = max(0, box_x - margin), max(0, box_y - margin)
        x2, y2 = min(padded_width, box_x + box_width + margin), min(padded_height, box_y + box_height + margin)
        cropped_padded = padded_inv[y1:y2, x1:x2]
    else:
        cropped_padded = inv_binary

    final_binary = cv2.bitwise_not(cropped_padded)

    # Step 7: Final white border & subtle blur.
    # PyTesseract struggles if characters are right on the edge of the crop.
    # We add 20px padding and apply a light 3x3 Gaussian blur to soften jagged text edges.
    padded_final = cv2.copyMakeBorder(final_binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    padded_final = cv2.GaussianBlur(padded_final, (3, 3), 0)

    return padded_final

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
    """
    try:
        # --psm 7: Treat the image as a single text line (perfect for license plates)
        # --oem 1: Use LSTM neural network OCR engine (more accurate than legacy engines)
        config = f"--psm 7 --oem 1"
        data_dict = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        
        words_confs = []
        for word, confidence in zip(data_dict.get('text', []), data_dict.get('conf', [])):
            # Clean up the output by removing special characters
            cleaned = PLATE_CHAR_PATTERN.sub('', str(word).strip().upper())
            if cleaned:
                try:
                    val = int(float(confidence))
                except (ValueError, TypeError):
                    val = -1
                # If confidence is missing (-1), default to a neutral 50%
                words_confs.append((cleaned, 50 if val == -1 else val))
        
        words = [word_confidence_pair[0] for word_confidence_pair in words_confs]
        confidences = [word_confidence_pair[1] for word_confidence_pair in words_confs]
        combined_text = " ".join(words)
        # Scale PyTesseract 0-100 confidence ratings down to a float between 0.0 and 1.0
        avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
        print(f"[OCR DEBUG] PyTesseract raw='{' '.join(data_dict.get('text', []))}', cleaned='{combined_text}', conf={avg_conf:.3f}")
        return combined_text, avg_conf
    except Exception as e:
        print(f"[OCR] PyTesseract execution failed: {str(e)}")
        return '', 0.0

def _run_easyocr_raw(processed):
    """
    Executes raw EasyOCR detection on a preprocessed image.
    """
    # Allowlist restricts characters to uppercase letters and digits.
    # Prevents OCR from misinterpreting letters/digits as punctuation (e.g. '/' or '-')
    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    # text_threshold=0.3: Minimum confidence score to accept characters.
    # low_text=0.2: Threshold for grouping nearby letters together.
    # mag_ratio=1.5: Upscales internal crop size to capture details in smaller text.
    results = get_easyocr_reader().readtext(processed, allowlist=allowlist, paragraph=False, text_threshold=0.3, low_text=0.2, mag_ratio=1.5)
    if not results:
        return '', 0.0

    # Sort results geometrically.
    # round(y / 20) groups characters into lines (vertical layout checks) and sorts them left-to-right (x-coordinate)
    results_sorted = sorted(results, key=lambda detection_result: (round(detection_result[0][0][1] / 20), detection_result[0][0][0]))
    combined_text = PLATE_CHAR_PATTERN.sub('', " ".join(detection_result[1].upper().strip() for detection_result in results_sorted))
    # Extract confidences (0.0 to 1.0 float returned by EasyOCR)
    confidences = [float(detection_result[2]) for detection_result in results_sorted if len(detection_result) > 2]
    avg_conf = np.mean(confidences) if confidences else 0.0
    return combined_text, avg_conf

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
    
    If using PyTesseract, runs both adaptive and Otsu binarization and selects the best read.
    """
    if engine_name != "PyTesseract":
        processed = preprocess_for_easyocr(cropped_plate_img)
        if processed is None:
            return '', 0.0, engine_name, None
        text, conf = _run_ocr(processed, engine_name)
        return text, conf, engine_name, processed

    candidates = []
    # Test both adaptive and Otsu thresholding for Tesseract since plate backgrounds
    # (dark/light plates, reflective coatings, shadows) respond differently to each method.
    for thresh_method in ("adaptive", "otsu"):
        processed = preprocess_for_tesseract(cropped_plate_img, thresh_method)
        if processed is not None:
            text, conf = _run_ocr(processed, engine_name)
            # Accept immediately if confidence is high (>= 0.50) to minimize runtime
            if text and conf >= 0.50:
                return text, conf, engine_name, processed
            candidates.append((text, conf, processed))

    valid = [c for c in candidates if c[0].strip()]
    if valid:
        # Fall back to returning the candidate that achieved the highest confidence score
        best = max(valid, key=lambda x: x[1])
        return best[0], best[1], engine_name, best[2]
        
    return '', 0.0, engine_name, None
