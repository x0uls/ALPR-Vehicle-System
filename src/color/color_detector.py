import cv2
import numpy as np

def _hue_to_color(hue):
    """
    Translates a numerical HSV Hue value (0-179 in OpenCV) into a plain-English color name.
    
    Hue corresponds to the 'tint' or location on the color wheel.
    """
    # Red wraps around the beginning and end of the HSV spectrum (0-8 and 165-179)
    if hue < 8 or hue >= 165:
        return "red"
    # Orange spans a narrow region between red and yellow (8 to 21)
    if hue < 22:
        return "orange"
    # Yellow covers the range from 22 to 37
    if hue < 38:
        return "yellow"
    # Green covers a wide band of mid-spectrum colors (38 to 84)
    if hue < 85:
        return "green"
    # Blue covers the cool color band (85 to 134)
    if hue < 135:
        return "blue"
    # Anything higher up to the red wraparound is purple
    return "purple"


def _classify_achromatic(weights, brightness_values):
    """
    Classifies non-colorful (achromatic) clusters into black, gray, silver, or white.
    
    Uses cluster weights (frequency of pixels) and brightness_values to decide.
    """
    # If there are no non-colorful pixels detected, fallback to a neutral gray
    if weights.sum() == 0:
        return "gray"

    # If 70% or more of the pixels have a brightness below 65, classify as black
    if weights[brightness_values < 65].sum() >= 0.70:
        return "black"

    # Paint mask filters out dark pixels (brightness < 65) to isolate the actual paint
    # rather than shadows, dark tires, or windows
    paint_mask = brightness_values >= 65
    if not paint_mask.any():
        return "black"

    paint_weights, paint_brightness_values = weights[paint_mask], brightness_values[paint_mask]

    # If at least 20% of the paint pixels are extremely bright (brightness >= 200), classify as white.
    # This helps identify white cars even when they are partially in shadows
    if paint_weights[paint_brightness_values >= 200].sum() >= 0.20:
        return "white"

    # Compute the average brightness of the paint pixels, weighted by cluster size
    weighted_brightness = np.average(paint_brightness_values, weights=paint_weights)
    
    # Brightness > 185 represents clean white paint under general daylight
    if weighted_brightness > 185:
        return "white"
    # Brightness > 125 represents reflective metallic paint (silver)
    if weighted_brightness > 125:
        return "silver"
    # Low-brightness paint is classified as neutral gray
    return "gray"


def detect_dominant_color(cropped_vehicle_img):
    """
    Extracts and classifies the dominant paint color of a vehicle bounding box.
    
    Uses K-Means clustering to group pixels, filters out non-paint regions, 
    and classifies the main color as either chromatic (colorful) or achromatic (grayscale).
    """
    if cropped_vehicle_img is None or cropped_vehicle_img.size == 0:
        return "gray"

    # 1. Crop to the center-body region of the vehicle to avoid road shadows (bottom),
    # windshield glass/interior (top), and surrounding background (left/right).
    height, width = cropped_vehicle_img.shape[:2]
    crop_y_start, crop_y_end = int(height * 0.45), int(height * 0.85)  # Vertical slice: middle to lower body
    crop_x_start, crop_x_end = int(width * 0.20), int(width * 0.80)  # Horizontal slice: center 60% of width
    body_region = cropped_vehicle_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    if body_region.size == 0:
        body_region = cropped_vehicle_img

    # 2. Resize the body image to 50x50 to speed up K-Means clustering, then flatten it
    # to a list of 2500 pixel rows with 3 channels (BGR) represented as floats.
    pixels = cv2.resize(body_region, (50, 50)).reshape(-1, 3).astype(np.float32)

    # 3. Apply K-Means clustering to find K=3 dominant color clusters (centers).
    # K=3 is chosen to separate paint color, shadows/under-car dark spots, and bright highlights.
    cluster_count = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, cluster_count, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # 4. Calculate the weight (percentage of pixels) belonging to each of the 3 clusters.
    weights = np.bincount(labels.flatten(), minlength=cluster_count) / len(labels)
    
    # 5. Convert the cluster centers from BGR to HSV color space to make color analysis easier
    hsv_centers = cv2.cvtColor(np.uint8([centers]), cv2.COLOR_BGR2HSV)[0].astype(int)
    hue, saturations, brightness_values = hsv_centers[:, 0], hsv_centers[:, 1], hsv_centers[:, 2]

    # 6. Mark a cluster as achromatic (grayscale) if its saturation is low (< 48)
    # or if it is extremely dark (brightness < 50)
    is_achromatic = (saturations < 48) | (brightness_values < 50)
    chromatic_weight = weights[~is_achromatic].sum()

    # 7. If colorful (chromatic) clusters make up at least 15% of the vehicle body region,
    # we classify the car using the heaviest colorful cluster.
    if chromatic_weight >= 0.15:
        # Penalize achromatic clusters by setting their weights to -1 so np.argmax ignores them
        chromatic_weighted = np.where(is_achromatic, -1, weights)
        dominant_idx = np.argmax(chromatic_weighted)
        return _hue_to_color(hue[dominant_idx])

    # 8. Otherwise, the car is grayscale/neutral; classify it as black, gray, silver, or white.
    return _classify_achromatic(weights[is_achromatic], brightness_values[is_achromatic])
