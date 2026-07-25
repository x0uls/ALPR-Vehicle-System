import cv2
import numpy as np
from sklearn.cluster import KMeans

def _hue_to_color(hue):
    """
    Translates a numerical HSV Hue value (0-179 in OpenCV) into a plain-English color name.
    """
    if hue < 8 or hue >= 165:
        return "red"
    if hue < 22:
        return "orange"
    if hue < 38:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 135:
        return "blue"
    return "purple"


def _classify_achromatic(weights, brightness_values):
    """
    Classifies non-colorful (achromatic) clusters into black, gray, silver, or white.
    """
    if len(weights) == 0 or weights.sum() == 0:
        return "gray"

    # Fix: Use safe boolean indexing directly matching the sliced arrays passed in
    if weights[brightness_values < 65].sum() >= 0.70:
        return "black"

    paint_mask = brightness_values >= 65
    if not paint_mask.any():
        return "black"

    paint_weights = weights[paint_mask]
    paint_brightness_values = brightness_values[paint_mask]

    if paint_weights[paint_brightness_values >= 200].sum() >= 0.20:
        return "white"

    weighted_brightness = np.average(paint_brightness_values, weights=paint_weights)
    
    if weighted_brightness > 185:
        return "white"
    if weighted_brightness > 125:
        return "silver"
    return "gray"


def detect_dominant_color(cropped_vehicle_img):
    """Extracts and classifies the dominant paint color of a vehicle crop."""
    if cropped_vehicle_img is None or cropped_vehicle_img.size == 0:
        return "gray"

    height, width = cropped_vehicle_img.shape[:2]
    
    # 1. Focus crop inward to eliminate wheels, road, sky, and background noise
    crop_y_start, crop_y_end = int(height * 0.30), int(height * 0.70)
    crop_x_start, crop_x_end = int(width * 0.25), int(width * 0.75)
    body_region = cropped_vehicle_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    if body_region.size == 0:
        body_region = cropped_vehicle_img

    # 2. Downsample for rapid execution speed
    resized = cv2.resize(body_region, (50, 50))
    hsv_region = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # 3. Filter out deep shadows and high-intensity glares
    h_channel = hsv_region[:, :, 0]
    s_channel = hsv_region[:, :, 1]
    v_channel = hsv_region[:, :, 2]
    valid_mask = (v_channel >= 30) & ~((v_channel > 245) & (s_channel < 20))
    
    # FIX: Pull pixel data from hsv_region, NOT from resized BGR image
    valid_pixels = hsv_region[valid_mask].reshape(-1, 3).astype(np.float32)
    if len(valid_pixels) < 10:
        valid_pixels = hsv_region.reshape(-1, 3).astype(np.float32)

    # 4. Cluster color coordinates using K-Means directly in HSV space
    cluster_count = 3
    kmeans = KMeans(n_clusters=cluster_count, n_init=5, random_state=42).fit(valid_pixels)
    labels, centers = kmeans.labels_, kmeans.cluster_centers_

    weights = np.bincount(labels, minlength=cluster_count) / len(labels)
    
    # FIX: Centers are already HSV; round directly instead of converting BGR->HSV
    hsv_centers = np.uint8(centers).astype(int)
    hue, saturations, brightness_values = hsv_centers[:, 0], hsv_centers[:, 1], hsv_centers[:, 2]

    # 5. Segment into Chromatic vs Achromatic colors
    is_achromatic = (saturations < 48) | (brightness_values < 50)
    chromatic_weight = weights[~is_achromatic].sum()

    # If we have enough vibrant color data, trust the chromatic hue
    if chromatic_weight >= 0.15:
        chromatic_weighted = np.where(is_achromatic, -1, weights)
        dominant_idx = np.argmax(chromatic_weighted)
        return _hue_to_color(hue[dominant_idx])

    # FIX: Fallback to avoid empty tracking mask indexing exceptions
    if not is_achromatic.any():
        # Edge case: Everything was labeled chromatic but fell short of the 0.15 threshold total weight
        dominant_idx = np.argmax(weights)
        return _hue_to_color(hue[dominant_idx])

    return _classify_achromatic(weights[is_achromatic], brightness_values[is_achromatic])
