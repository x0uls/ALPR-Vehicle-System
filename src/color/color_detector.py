import cv2
import numpy as np

def detect_dominant_color(cropped_vehicle_img):
    """
    Classify the vehicle's paint color using robust, vectorized HSV pixel-binning.
    
    This avoids the pitfalls of global median calculations and double loops.
    """
    if cropped_vehicle_img is None or cropped_vehicle_img.size == 0:
        return "gray"

    h, w = cropped_vehicle_img.shape[:2]

    # Sample the central/lower body area (avoids windows and tires/road)
    y1, y2 = int(h * 0.45), int(h * 0.85)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    body_region = cropped_vehicle_img[y1:y2, x1:x2]

    if body_region.size == 0:
        body_region = cropped_vehicle_img

    # Downsample to speed up calculation and smooth high-frequency noise
    resized = cv2.resize(body_region, (50, 50))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    h_vals, s_vals, v_vals = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Achromatic check: low saturation or extreme darkness
    achromatic = (s_vals < 45) | (v_vals < 50)
    black_mask = achromatic & (v_vals < 65)
    white_mask = achromatic & (~black_mask) & (v_vals > 195)
    silver_mask = achromatic & (~black_mask) & (~white_mask) & (v_vals > 135)
    gray_mask = achromatic & (~black_mask) & (~white_mask) & (~silver_mask)

    # Chromatic check: classify by Hue (0 - 180 range in OpenCV)
    chromatic = ~achromatic
    red_mask = chromatic & ((h_vals < 8) | (h_vals >= 165))
    orange_mask = chromatic & (h_vals >= 8) & (h_vals < 22)
    yellow_mask = chromatic & (h_vals >= 22) & (h_vals < 38)
    green_mask = chromatic & (h_vals >= 38) & (h_vals < 85)
    blue_mask = chromatic & (h_vals >= 85) & (h_vals < 135)
    purple_mask = chromatic & (h_vals >= 135) & (h_vals < 165)

    color_counts = {
        "black": np.sum(black_mask),
        "white": np.sum(white_mask),
        "silver": np.sum(silver_mask),
        "gray": np.sum(gray_mask),
        "red": np.sum(red_mask),
        "orange": np.sum(orange_mask),
        "yellow": np.sum(yellow_mask),
        "green": np.sum(green_mask),
        "blue": np.sum(blue_mask),
        "purple": np.sum(purple_mask)
    }

    # Sum up chromatic pixels
    chromatic_keys = ["red", "orange", "yellow", "green", "blue", "purple"]
    total_chromatic = sum(color_counts[k] for k in chromatic_keys)
    total_pixels = 50 * 50

    # Decision logic
    if total_chromatic / total_pixels >= 0.10:
        return max(chromatic_keys, key=lambda k: color_counts[k])
    else:
        achromatic_keys = ["black", "white", "silver", "gray"]
        return max(achromatic_keys, key=lambda k: color_counts[k])
