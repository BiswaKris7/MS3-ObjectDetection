import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as transforms

# Load the YOLOv8 object detection model
yolo_model = YOLO("yolov8n.pt")

# Load the MiDaS depth estimation model
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
depth_model.eval()

# Define image transformation for MiDaS model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),  # Resize to MiDaS input size
    transforms.ToTensor()
])

# Define dataset path
dataset_folder = "Dataset_Occluded_Pedestrian"

# Get the first available image
image_files = [f for f in os.listdir(dataset_folder) if f.endswith(".png")]
if not image_files:
    print("❌ No images found in the dataset.")
    exit()

image_path = os.path.join(dataset_folder, image_files[0])
print(f"🟢 Loading image: {image_path}")

# Read the image
image = cv2.imread(image_path)
if image is None:
    print(f"❌ Failed to load {image_path}")
    exit()

# Get original image dimensions
H, W, _ = image.shape

# Run YOLO object detection
yolo_results = yolo_model(image)

# Prepare image for depth estimation
depth_input = transform(image).unsqueeze(0)

# Run depth estimation
with torch.no_grad():
    depth_map = depth_model(depth_input)
depth_map = depth_map.squeeze().numpy()

# Normalize depth values for visualization
depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

# Resize depth map to match original image size
depth_map = cv2.resize(depth_map, (W, H))

# Debugging: Print depth map information
print(f"🔍 Depth Map Shape: {depth_map.shape}")
print(f"🔍 Depth Min: {np.min(depth_map)}, Max: {np.max(depth_map)}")

# Process YOLO results
for result in yolo_results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class index

        # Only process pedestrians (class index 0 = person)
        if cls == 0:
            # Ensure bounding box is within valid image bounds
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W - 1))
            y2 = max(0, min(y2, H - 1))

            # Ensure bounding box has a valid width and height
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                print(f"❌ Skipping invalid pedestrian box: ({x1}, {y1}) -> ({x2}, {y2}) [Too small or zero-width/height]")
                continue

            # Crop depth map to pedestrian bounding box
            pedestrian_depth = depth_map[y1:y2, x1:x2]

            # If the pedestrian depth map is empty, skip it
            if pedestrian_depth.size == 0:
                print(f"❌ Skipping empty depth map for pedestrian at ({x1}, {y1}) -> ({x2}, {y2})")
                continue

            avg_depth = np.mean(pedestrian_depth)  # Compute average depth

            # Draw bounding box and display depth info
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {conf:.2f}, Depth: {avg_depth:.2f}m"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the output
cv2.imshow("YOLOv8 Detection + Depth Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
