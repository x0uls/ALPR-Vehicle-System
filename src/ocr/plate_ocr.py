import os
os.environ["OMP_THREAD_LIMIT"] = "1"
import re
import cv2
import numpy as np
import easyocr
import torch
import pytesseract

for t_path in ["/usr/bin/tesseract", "/usr/local/bin/tesseract", r"C:\Program Files\Tesseract-OCR\tesseract.exe"]:
    if os.path.exists(t_path):
        pytesseract.pytesseract.tesseract_cmd = t_path
        break

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

PLATE_CHAR_PATTERN = re.compile(r'[^A-Z0-9]')
MIN_PLATE_LENGTH = 3
MAX_PLATE_LENGTH = 10
_TESS_CONFIG = "--psm 7 --oem 1"


def _fix_plate_format(text):
    compact = PLATE_CHAR_PATTERN.sub('', text.strip().upper()).replace(' ', '')
    if len(compact) < MIN_PLATE_LENGTH: return text

    d2l, l2d = str.maketrans('01258', 'OIZSB'), str.maketrans('OIZSB', '01258')
    chars = list(compact)
    if chars[0].isdigit(): chars[0] = chars[0].translate(d2l)

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

    result = "".join(chars)
    for i in range(1, len(result)):
        if result[i-1].isalpha() and result[i].isdigit():
            return result[:i] + ' ' + result[i:]
    return result


def _add_white_padding(img, pad=15):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255) if len(img.shape)==3 else 255)


def _preprocess_clean_2x(crop):
    h, w = crop.shape[:2]
    up = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return _add_white_padding(up, pad=10)


def _preprocess_tesseract_shaved_binarized(crop):
    h, w = crop.shape[:2]
    shave_h, shave_w = int(h * 0.08), int(w * 0.10)
    if shave_h > 0 and shave_w > 0:
        crop = crop[shave_h:h-shave_h, shave_w:w-shave_w]

    h, w = crop.shape[:2]
    up = cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if len(up.shape) == 3 else up
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    return cv2.cvtColor(_add_white_padding(thresh, pad=15), cv2.COLOR_GRAY2RGB)


def _postprocess_ocr_text(combined_text, avg_conf):
    compact = PLATE_CHAR_PATTERN.sub('', combined_text.strip().upper()).replace(' ', '')
    if not (MIN_PLATE_LENGTH <= len(compact) <= MAX_PLATE_LENGTH):
        return '', 0.0
    return _fix_plate_format(compact), max(0.01, avg_conf * min(1.0, len(compact) / 7.0))


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
        return "".join(words), (float(np.mean(confs)) / 100.0) if confs else 0.0
    except Exception as e:
        print(f"[PyTesseract WARN] {e}")
        return '', 0.0


def _run_easyocr_raw(processed):
    try:
        results = reader.readtext(processed)
        if not results: return '', 0.0
        sorted_results = sorted(results, key=lambda r: r[0][0][0])
        combined_text = "".join([r[1].upper().strip() for r in sorted_results])
        cleaned = PLATE_CHAR_PATTERN.sub('', combined_text)
        confs = [float(r[2]) for r in sorted_results if len(r) > 2]
        return cleaned, float(np.mean(confs)) if confs else 0.0
    except Exception as e:
        print(f"[EasyOCR WARN] {e}")
        return '', 0.0


def read_plate(cropped_plate_img, engine_name="EasyOCR"):
    if cropped_plate_img is None or cropped_plate_img.size == 0:
        return '', 0.0, engine_name, None

    try:
        if engine_name == "PyTesseract":
            proc = _preprocess_tesseract_shaved_binarized(cropped_plate_img)
            raw_text, conf = _run_pytesseract_raw(proc)
        else:
            proc = _preprocess_clean_2x(cropped_plate_img)
            raw_text, conf = _run_easyocr_raw(proc)

        final_text, final_conf = _postprocess_ocr_text(raw_text, conf)
        return final_text, final_conf, engine_name, proc
    except Exception as e:
        print(f"[OCR WARN] {engine_name}: {e}")
        return '', 0.0, engine_name, None
