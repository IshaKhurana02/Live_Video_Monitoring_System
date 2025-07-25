import numpy as np
import cv2
import json
import torch
from torchvision import transforms
from collections import deque
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from csrnet_pytorch.model import CSRNet

class_id_to_attr = {
    0: 'Female', 1: 'AgeOver60', 2: 'Age18-60', 3: 'AgeLess18', 4: 'Front',
    5: 'Side', 6: 'Back', 7: 'Hat', 8: 'Glasses', 9: 'HandBag',
    10: 'ShoulderBag', 11: 'Backpack', 12: 'HoldObjectsInFront', 13: 'ShortSleeve',
    14: 'LongSleeve', 15: 'UpperStride', 16: 'UpperLogo', 17: 'UpperPlaid',
    18: 'UpperSplice', 19: 'LowerStripe', 20: 'LowerPattern', 21: 'LongCoat',
    22: 'Trousers', 23: 'Shorts', 24: 'Skirt&Dress', 25: 'boots'
}

def load_config(config_path='config.json'):
    with open(config_path,'r') as config_file:
        return json.load(config_file)
    
def detect_dominant_color_hsv(image, bbox):
    try:
        x1, y1, x2, y2 = map(int, bbox)
        # Ensure bounding box is within image dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return {"hsv": [0, 0, 0], "color_name": "Unknown"}

        # Extract ROI
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return {"hsv": [0, 0, 0], "color_name": "Unknown"}

        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Compute histogram for hue channel (0-179 in OpenCV)
        hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist)

        # Approximate color name based on hue
        color_ranges = {
            "Red": (0, 10),  # Includes 170-179 for red wrap-around
            "Orange": (11, 25),
            "Yellow": (26, 40),
            "Green": (41, 80),
            "Cyan": (81, 100),
            "Blue": (101, 130),
            "Purple": (131, 160),
            "Magenta": (161, 179)
        }

        color_name = "Unknown"
        for name, (lower, upper) in color_ranges.items():
            if lower <= dominant_hue <= upper or (name == "Red" and dominant_hue >= 170):
                color_name = name
                break

        # Average saturation and value for completeness
        mean_hsv = np.mean(hsv_roi, axis=(0, 1)).astype(int)
        return color_name
    except Exception as e:
        print(f"Error in detect_dominant_color_hsv: {e}")
        return "Unknown"
    
def load_person_attributes_resnet_model(model_path, device):
    
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 26),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_attributes(resnet_model, image, bbox, device):
    try:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return {}

        # Extract ROI and preprocess
        roi = image[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(roi_rgb).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image_pil).unsqueeze(0).to(device)

        # Predict attributes
        with torch.no_grad():
            outputs = resnet_model(input_tensor)
            probabilities = outputs.squeeze(0).cpu().numpy()

        # Attribute mapping
        attribute_probs = {class_id_to_attr[i]: probabilities[i] for i in range(len(probabilities))}

        # Categorize attributes
        attributes = {}

        # 1. Gender
        attributes["Gender"] = "Female" if attribute_probs["Female"] > 0.6 else "Male"

        # 2. Age
        age_probs = {
            "Over60": attribute_probs["AgeOver60"],
            "18-60": attribute_probs["Age18-60"],
            "Less18": attribute_probs["AgeLess18"]
        }
        attributes["Age"] = max(age_probs, key=age_probs.get)

        # 3. View
        view_probs = {
            "Front": attribute_probs["Front"],
            "Side": attribute_probs["Side"],
            "Back": attribute_probs["Back"]
        }
        attributes["View"] = max(view_probs, key=view_probs.get)

        # 4. Bag
        bag_probs = {
            "HandBag": attribute_probs["HandBag"],
            "ShoulderBag": attribute_probs["ShoulderBag"],
            "Backpack": attribute_probs["Backpack"]
        }
        attributes["Bag"] = max(bag_probs, key=bag_probs.get) if max(bag_probs.values()) > 0.5 else "None"

        # 5. Sleeves
        sleeve_probs = {
            "ShortSleeve": attribute_probs["ShortSleeve"],
            "LongSleeve": attribute_probs["LongSleeve"]
        }
        attributes["Sleeves"] = max(sleeve_probs, key=sleeve_probs.get)

        # 6. Upper Body
        upper_probs = {
            "UpperStride": attribute_probs["UpperStride"],
            "UpperLogo": attribute_probs["UpperLogo"],
            "UpperPlaid": attribute_probs["UpperPlaid"],
            "UpperSplice": attribute_probs["UpperSplice"]
        }
        attributes["UpperBody"] = max(upper_probs, key=upper_probs.get) if max(upper_probs.values()) > 0.5 else "None"

        # 7. Lower Body
        lower_probs = {
            "LowerStripe": attribute_probs["LowerStripe"],
            "LowerPattern": attribute_probs["LowerPattern"],
            "LongCoat": attribute_probs["LongCoat"],
            "Trousers": attribute_probs["Trousers"],
            "Shorts": attribute_probs["Shorts"],
            "Skirt&Dress": attribute_probs["Skirt&Dress"]
        }
        attributes["LowerBody"] = max(lower_probs, key=lower_probs.get) if max(lower_probs.values()) > 0.5 else "None"

        return attributes
    except Exception as e:
        print(f"Error in extract_attributes: {e}")
        return {}
    
def load_csrnet_model(model_path, device):
    model = CSRNet(load_weights=True)
    
    # Load the pre-trained weights file
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the weights are stored under the 'state_dict' key
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def csrnet_preprocess_frame(frame, input_size=(640, 640)):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL image
        transforms.Resize(input_size),  # Resize to the input size expected by the model
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])
    return transform(frame).unsqueeze(0)

def crowd_estimation_with_csrnet(frame, model, device):
  
    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]
    
    # Preprocess the frame
    input_tensor = csrnet_preprocess_frame(frame).to(device)
    
    # Generate density map
    with torch.no_grad():
        density_map = model(input_tensor)
    
    # Convert density map to numpy array
    density_map = density_map.squeeze().cpu().numpy()
    
    # Total crowd count (sum of density map)
    total_estimated_count = np.sum(density_map)
    
    return int(total_estimated_count), density_map