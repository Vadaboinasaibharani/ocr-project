# ocr_module.py
import cv2
import numpy as np
import re
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    _EASYOCR_AVAILABLE = False

try:
    import pytesseract
    _PYTESSERACT_AVAILABLE = True
except Exception:
    _PYTESSERACT_AVAILABLE = False

_EASY_READER = None
def _get_easy_reader():
    global _EASY_READER
    if not _EASY_READER and _EASYOCR_AVAILABLE:
        # gpu=False (CPU). If you have GPU-configured torch, set gpu=True
        _EASY_READER = easyocr.Reader(['en'], gpu=False)
    return _EASY_READER

def detect_display_bbox(img):
    """Find bright/green-ish display region; fallback to centered crop."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 20, 20])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        best, best_area = None, 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 400: continue
            ar = w / float(h)
            if 1.2 <= ar <= 12 and area > best_area:
                best = (x, y, w, h); best_area = area
        if best:
            return best
    H, W = img.shape[:2]
    w = int(W * 0.7); h = int(H * 0.25)
    x = (W - w) // 2; y = (H - h) // 2
    return (x, y, w, h)

def preprocess_roi(roi, upscale=2):
    if upscale != 1:
        roi = cv2.resize(roi, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)
    th = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 9)
    return enhanced, th

def ocr_easyocr_image(roi):
    reader = _get_easy_reader()
    if reader is None:
        return []
    # easyocr expects RGB
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    out = reader.readtext(rgb, detail=1, paragraph=False)
    # out entries: [ (bbox, text, prob), ... ]
    return out

def ocr_tesseract_image(img_gray):
    if not _PYTESSERACT_AVAILABLE:
        return []
    data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT, config='--psm 6')
    out = []
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if not txt:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        conf_val = data['conf'][i]
        try:
            conf = float(conf_val)
        except:
            conf = -1.0
        bbox = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        out.append((bbox, txt, conf / 100.0))
    return out

def clean_token(tok):
    tok = tok.strip().replace(',', '.')
    if re.search(r'\d', tok):
        tok = tok.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('B', '8')
    return tok

def reconstruct_from_bboxes(results):
    tokens = []
    for (bbox, text, prob) in results:
        try:
            xs = [p[0] for p in bbox]
            cx = int(sum(xs) / len(xs))
        except:
            cx = 0
        tokens.append((text, cx))
    tokens.sort(key=lambda t: t[1])
    return tokens

def parse_numeric(tokens, joined_raw):
    joined_raw = joined_raw.replace(',', '.')
    m = re.search(r'\d+\.\d+', joined_raw)
    if m:
        return float(m.group())
    nums = [clean_token(t) for t, _ in tokens if re.search(r'\d', t)]
    if not nums:
        return None
    joined = ''.join(nums)
    if '.' not in joined and len(joined) >= 4:
        val = float(joined[:-2] + '.' + joined[-2:])
        return round(val, 4)
    try:
        return float(joined)
    except:
        return None

def correct_numeric(value, label):
    if value is None:
        return None
    if label == 'Pressure' and value > 20:
        for d in (1000, 100, 10):
            v = value / d
            if 0 < v < 20:
                return round(v, 4)
    if label == 'Level' and value > 100:
        for d in (100, 10):
            v = value / d
            if 0 < v <= 100:
                return round(v, 4)
    return round(value, 4)

def detect_label(joined, value):
    text = joined.lower()
    if any(k in text for k in ['he level', 'heleve', 'level', 'leve1', '%']):
        return 'Level'
    if any(k in text for k in ['he pressure', 'pressure', 'psi']):
        return 'Pressure'
    if value is not None:
        if value <= 100:
            return 'Level'
        elif value > 100:
            return 'Pressure'
    return None

def analyze_image(path, show=False):
    """Return dict: {'path', 'label', 'value', 'raw_text', 'annotated_path'}"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    x, y, w, h = detect_display_bbox(img)
    roi = img[y:y + h, x:x + w]
    enhanced, th = preprocess_roi(roi)

    # Try EasyOCR first, fallback to pytesseract
    results = []
    if _EASYOCR_AVAILABLE:
        try:
            eout = ocr_easyocr_image(roi)
            # easyocr returns bbox as list of 4 points
            for bbox, txt, prob in eout:
                results.append((bbox, txt, prob))
        except Exception:
            results = []
    if not results and _PYTESSERACT_AVAILABLE:
        try:
            tout = ocr_tesseract_image(th)
            results = tout
        except Exception:
            results = []

    joined = " | ".join([r[1] for r in results])
    tokens = reconstruct_from_bboxes(results)
    value = parse_numeric(tokens, joined)
    label = detect_label(joined, value)
    value = correct_numeric(value, label)

    # Annotate on original image (for display)
    annotated = img.copy()
    for (bbox, text, _) in results:
        try:
            tl = (int(bbox[0][0]), int(bbox[0][1]))
            br = (int(bbox[2][0]), int(bbox[2][1]))
        except:
            # fallback
            tl = (0, 0); br = (0, 0)
        cv2.rectangle(annotated, tl, br, (0, 128, 255), 2)
        cv2.putText(annotated, str(text), (tl[0], max(10, tl[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

    # Save annotated image to temp path
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    annotated_path = tf.name
    tf.close()
    cv2.imwrite(annotated_path, annotated)

    return {
        'path': path,
        'label': label,
        'value': value,
        'raw_text': joined,
        'annotated_path': annotated_path
    }
