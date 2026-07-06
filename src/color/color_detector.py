import cv2
import numpy as np


def detect_dominant_color(cropped_vehicle_img, k=3):
    """
    Academic Rationale (Unsupervised Machine Learning - Clustering):
    To determine the vehicle's color, we use K-Means clustering, an unsupervised machine learning algorithm.
    The vehicle crop image contains thousands of pixels (colors). We flatten these pixels and ask K-Means 
    to group them into k dominant clusters. We mask out the darkest/lightest/desaturated clusters (often shadows, 
    tires, or glare) and extract the most prominent valid colored cluster as the dominant vehicle paint color.

    Two algorithmic optimizations over a naive whole-box approach:
    1. Central Sampling: Only sample the central region of the bounding box (avoiding background edges and roof/windshield).
    2. HSV Hue Classification: Hue is mathematically stable under lighting changes (shadows drop brightness but keep hue constant).
    """
    h, w = cropped_vehicle_img.shape[:2]

    # Sample the central/lower body area. This avoids box-edge
    # background, roof/windshield reflections, and small trim details.
    y1, y2 = int(h * 0.42), int(h * 0.88)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    body_region = cropped_vehicle_img[y1:y2, x1:x2]

    if body_region.size == 0:
        body_region = cropped_vehicle_img

    resized = cv2.resize(body_region, (50, 50))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    pixels_hsv = hsv.reshape((-1, 3))

    h_ch, s_ch, v_ch = pixels_hsv[:, 0], pixels_hsv[:, 1], pixels_hsv[:, 2]

    # Mask out pixels that don't represent a clear paint color:
    # - very dark (shadows, tinted windows, tires)
    # - very bright with low saturation (specular highlights/glare)
    # - low saturation overall (grays don't have a usable hue)
    valid_mask = (v_ch > 45) & (v_ch < 245) & (s_ch > 55)

    valid_pixels_bgr = resized.reshape((-1, 3))[valid_mask]

    # Pre-calculate the fallback color using the uniform matrix shape
    fallback_color = classify_achromatic_from_bgr(resized.reshape((-1, 3)))

    color_coverage = len(valid_pixels_bgr) / len(pixels_hsv)
    if len(valid_pixels_bgr) < 10 or color_coverage < 0.18:
        return fallback_color

    pixels_for_kmeans = valid_pixels_bgr.astype(np.float32)
    k = min(k, len(pixels_for_kmeans))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels_for_kmeans, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    counts = np.bincount(labels.flatten())
    dominant_idx = int(np.argmax(counts))
    dominant_share = counts[dominant_idx] / counts.sum()
    
    # FIX: Avoid calling shape-sensitive logic with flattening structures
    if dominant_share * color_coverage < 0.18:
        return fallback_color

    dominant = centers[dominant_idx]
    b, g, r = dominant
    return classify_color(r, g, b)


def classify_achromatic_from_bgr(pixels_bgr):
    hsv = cv2.cvtColor(pixels_bgr.reshape((-1, 1, 3)).astype(np.uint8), cv2.COLOR_BGR2HSV)
    values = hsv[:, 0, 2]
    median_v = float(np.median(values))
    if median_v < 65:
        return "black"
    if median_v > 205:
        return "white"
    if median_v > 135:
        return "silver"
    return "gray"


def classify_color(r, g, b):
    """Classify an RGB color by HUE (HSV) rather than RGB Euclidean
    distance. RGB distance fails on shadowed/lit colors because
    brightness changes move a pixel a lot in RGB space without its
    actual color identity changing -- hue is far more stable.
    """
    bgr_pixel = np.uint8([[[b, g, r]]])
    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

    # Achromatic checks first (low saturation or extreme brightness
    # mean hue is meaningless/noisy for these)
    if v < 50:
        return "black"
    if s < 30:
        if v > 200:
            return "white"
        elif v > 130:
            return "silver"
        else:
            return "gray"

    # Hue-based ranges (OpenCV hue range is 0-180)
    if h < 8 or h >= 170:
        return "red"
    elif h < 20:
        return "orange"
    elif h < 35:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 130:
        return "blue"
    elif h < 155:
        return "purple"
    else:
        return "red"
