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
    
    # Initialize color counts using vectorized masking
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    
    # Achromatic checks (low saturation or extreme darkness)
    achromatic = (S < 45) | (V < 50)
    
    black_count = np.sum(achromatic & (V < 65))
    white_count = np.sum(achromatic & (V >= 65) & (V > 195))
    silver_count = np.sum(achromatic & (V >= 65) & (V <= 195) & (V > 135))
    gray_count = np.sum(achromatic & (V >= 65) & (V <= 135))
    
    # Chromatic checks
    chromatic = ~achromatic
    
    red_count = np.sum(chromatic & ((H < 8) | (H >= 165)))
    orange_count = np.sum(chromatic & (H >= 8) & (H < 165) & (H < 22))
    yellow_count = np.sum(chromatic & (H >= 22) & (H < 165) & (H < 38))
    green_count = np.sum(chromatic & (H >= 38) & (H < 165) & (H < 85))
    blue_count = np.sum(chromatic & (H >= 85) & (H < 165) & (H < 135))
    purple_count = np.sum(chromatic & (H >= 135) & (H < 165))
    
    chromatic_counts = {
        "red": red_count,
        "orange": orange_count,
        "yellow": yellow_count,
        "green": green_count,
        "blue": blue_count,
        "purple": purple_count
    }
    
    achromatic_counts = {
        "black": black_count,
        "white": white_count,
        "silver": silver_count,
        "gray": gray_count
    }
    
    total_chromatic = sum(chromatic_counts.values())
    total_pixels = 50 * 50
    chromatic_ratio = total_chromatic / total_pixels
    
    if chromatic_ratio >= 0.10:
        return max(chromatic_counts, key=chromatic_counts.get)
    else:
        return max(achromatic_counts, key=achromatic_counts.get)
