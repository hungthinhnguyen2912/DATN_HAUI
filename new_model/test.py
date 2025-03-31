import cv2
import os
from collections import Counter

DATASET_PATH = r"D:\datn_haui\new_model\dataset\fruits-360\Training"  # Use raw string
sizes = []

# Iterate over all class folders
for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):  # Skip if it's not a directory
        continue

    # Iterate over images inside the class folder
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        if not os.path.isfile(img_path):  # Skip if it's not a file
            print(f"❌ Missing file: {img_path}")
            continue

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image
        if image is None:
            print(f"❌ Cannot read file: {img_path}")
            continue

        sizes.append(image.shape[:2])  # Store (height, width)

# Check if any images were read
if not sizes:
    print("🚨 No valid images found! Check dataset path and image formats.")
else:
    size_counts = Counter(sizes)
    most_common_size = size_counts.most_common(1)[0]
    print(f"📏 Most common size: {most_common_size[0]} (Appears {most_common_size[1]} times)")
