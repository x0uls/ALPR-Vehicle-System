import os
# Limits OpenMP (a multi-threading library used internally by OpenCV/PyTorch) to 1 thread.
# Why: this pipeline is already run inside its own parallel thread pools (one for EasyOCR,
# one for PyTesseract). If OpenMP is ALSO allowed to spawn multiple threads inside each of
# those, you get "thread oversubscription" (way more threads than CPU cores), which can
# cause slowdowns, contention, or in some cases crashes. Capping it to 1 avoids that.
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import time
import cv2
import numpy as np
import easyocr
import torch
import pytesseract

# Automatically locate Tesseract-OCR executable binary across Linux and Windows default paths.
# pytesseract is just a Python wrapper — it needs to know where the actual Tesseract
# program is installed on disk to call it.
for t_path in ["/usr/bin/tesseract", "/usr/local/bin/tesseract", r"C:\Program Files\Tesseract-OCR\tesseract.exe"]:
    if os.path.exists(t_path):
        pytesseract.pytesseract.tesseract_cmd = t_path
        break

# Initialize EasyOCR reader model instance ONCE at import time (not per-request).
# Why: loading the neural network weights is slow — doing it once and reusing the
# same `reader` object for every image is much faster than reloading it every call.
# gpu=torch.cuda.is_available() automatically uses your GPU if one is detected, else falls back to CPU.
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# ── Regex and constants used for cleaning/validating OCR output ──────────────
PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9]')  # Matches anything that ISN'T A-Z or 0-9, so it can be stripped out
MIN_PLATE_LENGTH = 3                           # Reject anything shorter than this — too short to be a real plate
MAX_PLATE_LENGTH = 10                          # Reject anything longer than this — likely garbage/misread
_TESS_CONFIG = "--psm 7 --oem 1"               # PSM 7 = treat the image as ONE line of text (not a paragraph)
                                                # OEM 1 = use Tesseract's neural net (LSTM) engine, not the legacy engine


def _fix_plate_format(text):
    """
    Applies domain-specific Malaysian / standard license plate formatting heuristics.
    OCR engines often confuse visually similar characters (0 vs O, 1 vs I, 8 vs B, etc.)
    depending on whether that position SHOULD be a letter or a number. This function
    uses the expected plate structure (letters first, then numbers) to fix those mistakes.

    Steps:
    1. Removes all non-alphanumeric noise characters.
    2. Corrects "inverted" readings, e.g. OCR misreads the plate back-to-front: "9393WSX" -> "WSX9393".
    3. Positional character translation:
       - In letter positions: converts number-lookalikes to their letter equivalent (e.g. '0'->'O', '1'->'I', '5'->'S').
       - In number positions: converts letter-lookalikes to their digit equivalent (e.g. 'O'->'0', 'I'->'1', 'B'->'8').
    4. Inserts a space between the letter prefix and number suffix (standard plate formatting, e.g. "WD586D" -> "WD 586D").
    """
    compact = PLATE_CHAR_PATTERN.sub('', text.strip().upper()).replace(' ', '')
    if len(compact) < MIN_PLATE_LENGTH:
        return text  # too short to safely reformat — return as-is rather than risk mangling it further

    # Handle inverted text reading (e.g., "9393WSX" -> "WSX9393") for 3-4 digits followed by 2-3 letters.
    # This happens when OCR reads the plate's number/letter blocks in the wrong order.
    if compact[0].isdigit():
        m = re.match(r'^(\d{3,4})([A-Z]{2,3})$', compact)
        if m:
            digits, letters = m.group(1), m.group(2)
            compact = letters + digits

    # Translation tables (str.maketrans builds a character-to-character lookup map)
    d2l = str.maketrans('01258', 'OIZSB')  # Digit -> Letter: used when a position SHOULD be a letter
    l2d = str.maketrans('OIZSB', '01258')  # Letter -> Digit: used when a position SHOULD be a number

    chars = list(compact)
    # The very first character of a Malaysian plate is always a letter (state/series code) —
    # so if OCR read it as a digit, force-correct it to its letter lookalike.
    if chars[0].isdigit():
        chars[0] = chars[0].translate(d2l)

    # Find where the numeric block starts and ends, so we know which characters
    # "should" be letters (prefix/suffix) vs which "should" be digits (the middle block).
    first_dig = next((i for i, c in enumerate(chars) if c.isdigit()), None)
    if first_dig is not None:
        last_dig = max(i for i, c in enumerate(chars) if c.isdigit())
        for i in range(len(chars)):
            if i < first_dig:
                # Before the number block: this position should be a letter
                if chars[i].isdigit(): chars[i] = chars[i].translate(d2l)
            elif i <= last_dig:
                # Inside the number block: this position should be a digit
                # Special case: 'J' doesn't map cleanly via the table, so it's manually mapped to '7'
                if chars[i].isalpha(): chars[i] = '7' if chars[i] == 'J' else chars[i].translate(l2d)
            else:
                # After the number block (trailing letter suffix, some plates have this): should be a letter
                if chars[i].isdigit(): chars[i] = chars[i].translate(d2l)

    # Insert a space right where the letters end and numbers begin, matching standard plate formatting
    result = "".join(chars)
    for i in range(1, len(result)):
        if result[i-1].isalpha() and result[i].isdigit():
            return result[:i] + ' ' + result[i:]
    return result


def _add_white_padding(img, pad=15):
    """
    Adds a uniform 15px white border around the image crop.
    Why: OCR engines (especially PyTesseract) misinterpret characters touching the image
    border as noise or stray lines, which can confuse detection. Adding white margin
    space around the edges gives characters "breathing room" and improves reads.
    """
    border_color = (255, 255, 255) if len(img.shape) == 3 else 255  # white in RGB (3-channel) or plain white value (grayscale)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border_color)


def _upscale_and_denoise_first(crop, target_h=140):
    """
    Step 1: Shared Rescaling & Denoising Pipeline (used by BOTH EasyOCR and PyTesseract paths).

    - Scale Factor: Upscales small plate crops to a target height of 140px, since most
      raw plate crops are far too small (often under 30px tall) for OCR to reliably read
      individual characters. More pixels per character = more detail for the model to work with.
    - Interpolation (cv2.INTER_LANCZOS4): A high-quality resizing algorithm (Lanczos sinc
      resampling) that produces smoother results than basic resizing, avoiding blocky/pixelated
      edges when stretching a small image up to a much larger size.
    - Denoising (cv2.bilateralFilter): Smooths out random background noise (e.g. sensor grain,
      compression artifacts) WITHOUT blurring sharp edges — this matters because a normal blur
      would also soften the character edges we need for accurate OCR.
        * d=7: size of the pixel neighborhood considered for each smoothing operation.
        * sigmaColor=75: how much color/intensity difference is treated as "noise to smooth"
          vs "a real edge to preserve."
        * sigmaSpace=75: how far (spatially) the smoothing effect reaches from each pixel.
    """
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop

    # Calculate how much to scale up to reach the 140px target height (minimum 3x zoom, even
    # if the crop is already close to 140px, to guarantee enough resolution for OCR)
    scale = max(3.0, target_h / float(h))
    up = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    try:
        smooth = cv2.bilateralFilter(up, d=7, sigmaColor=75, sigmaSpace=75)
        return smooth
    except Exception:
        # If denoising fails for any reason, fall back to the upscaled-but-unsmoothed image
        # rather than crashing the whole pipeline
        return up


def _preprocess_clean_2x(crop):
    """
    Pre-processing pipeline for EasyOCR.
    EasyOCR is a deep learning model that works well directly on RGB (color) images —
    it doesn't need grayscale conversion or binarization like PyTesseract does, since
    its neural network learns to interpret color/contrast patterns on its own.

    1. Upscales and denoises crop to 140px target height (shared step above).
    2. Adds white margin padding so characters aren't clipped at the image border.
    """
    highres = _upscale_and_denoise_first(crop, target_h=140)
    return _add_white_padding(highres, pad=15)


def _preprocess_tesseract_single_pass(crop):
    """
    Pre-processing pipeline optimized specifically for PyTesseract.
    Unlike EasyOCR, Tesseract's recognition accuracy depends HEAVILY on the input being
    high-contrast black-text-on-white-background — so this pipeline does much more
    aggressive processing to get the image into that ideal format.
    """
    if crop is None or crop.size == 0:
        return crop

    # 1. Upscale and denoise image to high resolution (same shared step as EasyOCR's pipeline)
    highres = _upscale_and_denoise_first(crop, target_h=140)

    # Convert RGB/BGR color image to 8-bit single-channel grayscale.
    # Tesseract works on grayscale/binary images, not color — color information isn't useful to it.
    gray = cv2.cvtColor(highres, cv2.COLOR_BGR2GRAY) if len(highres.shape) == 3 else highres

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Boosts LOCAL contrast (rather than globally), which helps characters stand out
    # even if the plate has uneven lighting/glare across its surface.
    # - clipLimit=2.0: caps how aggressively contrast is boosted, to avoid amplifying noise/glare too much.
    # - tileGridSize=(8, 8): splits the image into an 8x8 grid and adjusts contrast per-region,
    #   rather than applying one flat contrast setting to the whole image.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Mild Sharpening via 2D Convolution Kernel
    # A convolution kernel is a small grid of numbers "slid" across every pixel to compute
    # a new value based on that pixel and its neighbors. This particular kernel sharpens edges:
    # Kernel Matrix: [[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]]
    # - Center weight (+3.0): strongly boosts the pixel's own value.
    # - Surrounding weights (-0.5 top/bottom/left/right): subtracts neighboring pixel values,
    #   which exaggerates the difference between a pixel and its neighbors — this is what
    #   makes edges look "sharper" (higher contrast at boundaries).
    # - Kernel Sum: 3.0 - (4 * 0.5) = 1.0 ("unit gain") — this means the overall brightness
    #   of the image stays the same; we're only boosting edge contrast, not making the image darker/lighter.
    sharpen_k = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_k)

    # 4. Otsu Binarization (Thresholding)
    # Converts the grayscale image into pure black-and-white (every pixel becomes either
    # 0 or 255), which is the ideal input format for Tesseract's character recognition.
    # - cv2.GaussianBlur(..., (3,3), 0): a very light blur applied first, to smooth out tiny
    #   noise spots introduced by the sharpening step above, so they don't get mistakenly
    #   turned into little black specks during thresholding.
    # - cv2.THRESH_OTSU: automatically calculates the BEST brightness cutoff point to split
    #   pixels into black vs white, based on the image's own brightness distribution
    #   (rather than using one fixed threshold value for every image).
    blur = cv2.GaussianBlur(sharpened, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Polarity Check (Inversion)
    # Tesseract expects DARK characters on a LIGHT background (like normal printed text).
    # Some plates are naturally light-text-on-dark-background (common on certain plate styles),
    # which would confuse Tesseract if left as-is.
    # If the image's average pixel brightness is low (<127, meaning mostly dark/black), it means
    # the background is dark — so we invert black<->white to flip it into the expected format.
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # Add white margin padding (same reasoning as EasyOCR's pipeline), then convert back to
    # a 3-channel RGB image, since some downstream code/display tools expect 3 channels
    # even though the actual content is just black-and-white.
    return cv2.cvtColor(_add_white_padding(thresh, pad=15), cv2.COLOR_GRAY2RGB)


def _postprocess_ocr_text(combined_text, avg_conf):
    """
    Cleans up whatever raw text string the OCR engine returned, turning it into a
    final validated plate reading (or rejecting it if it doesn't look like a real plate).

    1. Removes non-alphanumeric characters (stray symbols/noise from misreads).
    2. Validates the character length falls in a sane range (3-10 chars) — anything
       outside that is almost certainly a bad read, not a real plate.
    3. Applies the plate formatting/character-correction rules (_fix_plate_format above).
    4. Calculates a length-adjusted confidence score: if the OCR engine returned a very
       short string, its confidence gets scaled down, since a short read is less likely
       to be a genuinely complete, correct plate number.
    """
    compact = PLATE_CHAR_PATTERN.sub('', combined_text.strip().upper()).replace(' ', '')
    if not (MIN_PLATE_LENGTH <= len(compact) <= MAX_PLATE_LENGTH):
        return '', 0.0  # reject: doesn't look like a valid plate length
    return _fix_plate_format(compact), max(0.01, avg_conf * min(1.0, len(compact) / 7.0))


def _run_pytesseract_raw(processed):
    """
    Runs PyTesseract OCR extraction on the pre-processed (binarized) plate crop.
    Returns (raw_text_string, average_confidence_score) — NOT yet cleaned/validated,
    that happens later in _postprocess_ocr_text.
    """
    try:
        # image_to_data returns detailed word-by-word results (not just plain text),
        # including a confidence score for each detected "word" (text fragment)
        data = pytesseract.image_to_data(processed, config=_TESS_CONFIG, output_type=pytesseract.Output.DICT)
        words, confs = [], []
        for word, conf in zip(data.get('text', []), data.get('conf', [])):
            cleaned = PLATE_CHAR_PATTERN.sub('', str(word).strip().upper())
            if cleaned:  # skip empty fragments
                try: val = int(float(conf))
                except (ValueError, TypeError): val = -1  # Tesseract sometimes returns "-1" or invalid confidence
                words.append(cleaned)
                confs.append(50 if val == -1 else val)  # default to 50% confidence if Tesseract gave an invalid value
        # Join all detected word fragments together, and average their confidence scores (converted to 0-1 scale)
        return "".join(words), (float(np.mean(confs)) / 100.0) if confs else 0.0
    except Exception as e:
        print(f"[PyTesseract WARN] {e}")
        return '', 0.0


def _run_easyocr_raw(processed):
    """
    Runs EasyOCR text extraction on the pre-processed (RGB) plate crop.
    Returns (raw_text_string, average_confidence_score).
    """
    try:
        # readtext() returns a list of (bounding_box, text, confidence) tuples —
        # EasyOCR may detect the plate as several separate text fragments rather than one string
        results = reader.readtext(processed)
        if not results: return '', 0.0

        # Sort fragments LEFT-TO-RIGHT by their bounding box's x-coordinate, so when we join
        # them together the character order matches how the plate actually reads
        # (EasyOCR doesn't guarantee it returns fragments in reading order by default)
        sorted_results = sorted(results, key=lambda r: r[0][0][0])
        combined_text = "".join([r[1].upper().strip() for r in sorted_results])
        cleaned = PLATE_CHAR_PATTERN.sub('', combined_text)
        confs = [float(r[2]) for r in sorted_results if len(r) > 2]  # r[2] is the confidence score for each fragment
        return cleaned, float(np.mean(confs)) if confs else 0.0
    except Exception as e:
        print(f"[EasyOCR WARN] {e}")
        return '', 0.0


def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    """
    Main entry point for plate OCR recognition — this is the function the rest of your
    system calls to actually read a plate crop.

    Dispatches the crop to whichever engine was requested (EasyOCR or PyTesseract),
    running that engine's specific preprocessing pipeline first, then the shared
    text-cleaning/validation step. Also measures how long the whole process took
    (used for your benchmark's latency comparison).

    Returns: (plate_text, confidence, engine_name, processed_image_crop, latency_ms)
      - plate_text: final cleaned/formatted plate string (or '' if rejected)
      - confidence: final confidence score (0.0-1.0)
      - engine_name: which engine was used ("EasyOCR" or "PyTesseract")
      - processed_image_crop: the preprocessed image that was actually fed to the OCR
        engine (useful for displaying "what the model saw" in your dashboard, like the
        side-by-side comparison view you built)
      - latency_ms: how long this single plate took to process, in milliseconds
    """
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return '', 0.0, engine_name, None, 0.0

    t_start = time.perf_counter()  # high-precision timer start
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
        # If anything in the pipeline throws an unexpected error, fail gracefully
        # (return an empty result) rather than crashing the whole benchmark run
        print(f"[OCR WARN] {engine_name}: {e}")
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return '', 0.0, engine_name, None, latency_ms