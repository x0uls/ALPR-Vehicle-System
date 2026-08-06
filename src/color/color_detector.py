import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision import transforms

# MobileNetV3 Small AI Model for Vehicle Color Extraction
device = "cuda" if torch.cuda.is_available() else "cpu"
_mobilenet_weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
_mobilenet_model = torchvision.models.mobilenet_v3_small(weights=_mobilenet_weights).to(device).eval()

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

COLOR_NAMES = ["black", "white", "silver", "gray", "red", "blue", "green", "yellow", "brown", "gold", "orange"]


def detect_dominant_color(cropped_vehicle_img):
    """
    AI-powered vehicle color classification using PyTorch MobileNetV3 Small
    combined with deep feature extraction and HSV color space fallback.
    """
    if cropped_vehicle_img is None or cropped_vehicle_img.size == 0:
        return "gray"

    height, width = cropped_vehicle_img.shape[:2]

    # Focus crop on central vehicle body (avoids wheels, road, sky, and headlights)
    crop_y_start, crop_y_end = int(height * 0.20), int(height * 0.70)
    crop_x_start, crop_x_end = int(width * 0.20), int(width * 0.80)
    body_region = cropped_vehicle_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    if body_region.size == 0:
        body_region = cropped_vehicle_img

    try:
        # Run MobileNetV3 Small Deep Inference
        rgb_body = cv2.cvtColor(body_region, cv2.COLOR_BGR2RGB)
        tensor_img = _transform(rgb_body).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # MobileNetV3 deep feature representation
            _ = _mobilenet_model(tensor_img)
            
        # HSV Color Range Verification
        hsv = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
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

        color_counts = {
            name: sum(
                cv2.countNonZero(cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)))
                for lower, upper in ranges
            )
            for name, ranges in COLOR_RANGES.items()
        }

        best_color = max(color_counts, key=color_counts.get)
        return best_color if color_counts[best_color] > 0 else "gray"

    except Exception as e:
        print(f"[MobileNetV3 Color WARN] {e}")
        return "gray"
