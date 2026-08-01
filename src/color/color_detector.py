import cv2
import numpy as np

def detect_dominant_color(cropped_vehicle_img):
    """
    Fast, simple, and accurate vehicle color detection using HSV color range masks.
    Eliminates K-Means overhead and provides deterministic vehicle color classification.
    """
    if cropped_vehicle_img is None or cropped_vehicle_img.size == 0:
        return "gray"

    height, width = cropped_vehicle_img.shape[:2]

    # Focus crop on central vehicle body (avoids wheels, road, sky, and tail lights)
    crop_y_start, crop_y_end = int(height * 0.25), int(height * 0.65)
    crop_x_start, crop_x_end = int(width * 0.20), int(width * 0.80)
    body_region = cropped_vehicle_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    if body_region.size == 0:
        body_region = cropped_vehicle_img

    hsv = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)

    # Standard vehicle paint color boundaries in HSV space (Lower, Upper)
    COLOR_RANGES = {
        "black":  [((0, 0, 0), (179, 255, 75))],
        "white":  [((0, 0, 195), (179, 35, 255))],
        "silver": [((0, 0, 135), (179, 45, 195))],
        "gray":   [((0, 0, 75), (179, 45, 135))],
        "red":    [((0, 70, 50), (10, 255, 255)), ((165, 70, 50), (179, 255, 255))],
        "gold":   [((10, 30, 80), (25, 150, 200))],
        "orange": [((10, 150, 180), (25, 255, 255))],
        "yellow": [((25, 50, 100), (35, 255, 255))],
        "green":  [((35, 50, 50), (85, 255, 255))],
        "blue":   [((85, 50, 50), (135, 255, 255))],
        "brown":  [((10, 30, 30), (25, 180, 100))]
    }

    color_counts = {}
    for color_name, ranges in COLOR_RANGES.items():
        mask_combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            mask_combined = cv2.bitwise_or(mask_combined, mask)
        color_counts[color_name] = cv2.countNonZero(mask_combined)

    best_color = max(color_counts, key=color_counts.get)
    return best_color if color_counts[best_color] > 0 else "gray"
