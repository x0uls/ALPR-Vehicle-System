import cv2
import numpy as np

def detect_dominant_color(cropped_vehicle_img):
    """
    Classify the vehicle's paint color using robust HSV pixel-binning.
    
    This avoids the pitfalls of global median calculations which get contaminated 
    by dark windows, license plate casings, and road shadows.
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
    
    # Initialize color bins
    color_counts = {
        "black": 0,
        "white": 0,
        "silver": 0,
        "gray": 0,
        "red": 0,
        "orange": 0,
        "yellow": 0,
        "green": 0,
        "blue": 0,
        "purple": 0
    }

    # Classify each pixel in the hsv grid
    for row in hsv:
        for pixel in row:
            h, s, v = int(pixel[0]), int(pixel[1]), int(pixel[2])
            
            # Achromatic check: low saturation or extreme darkness
            if s < 45 or v < 50:
                if v < 65:
                    color_counts["black"] += 1
                elif v > 195:
                    color_counts["white"] += 1
                elif v > 135:
                    color_counts["silver"] += 1
                else:
                    color_counts["gray"] += 1
            else:
                # Chromatic check: classify by Hue (0 - 180 range in OpenCV)
                if h < 8 or h >= 165:
                    color_counts["red"] += 1
                elif h < 22:
                    color_counts["orange"] += 1
                elif h < 38:
                    color_counts["yellow"] += 1
                elif h < 85:
                    color_counts["green"] += 1
                elif h < 135:
                    color_counts["blue"] += 1
                else:
                    color_counts["purple"] += 1

    total_pixels = 50 * 50
    
    # Sum up chromatic pixels
    chromatic_keys = ["red", "orange", "yellow", "green", "blue", "purple"]
    total_chromatic = sum(color_counts[k] for k in chromatic_keys)
    chromatic_ratio = total_chromatic / total_pixels

    # Decision logic
    if chromatic_ratio >= 0.10:
        # Find dominant chromatic paint color
        return max(chromatic_keys, key=lambda k: color_counts[k])
    else:
        # Find dominant achromatic paint color
        achromatic_keys = ["black", "white", "silver", "gray"]
        return max(achromatic_keys, key=lambda k: color_counts[k])
