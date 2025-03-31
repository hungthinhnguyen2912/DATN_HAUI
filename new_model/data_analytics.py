import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import cv2

DATA_DIR = r"D:\datn_haui\new_model\dataset\fruits-360\Training"

class_names = sorted(os.listdir(DATA_DIR))
num_classes = len(class_names)

label_encoder = LabelEncoder()
class_labels = label_encoder.fit_transform(class_names)

image_counts = {class_name: len(os.listdir(os.path.join(DATA_DIR, class_name))) for class_name in class_names}
encoded_counts = {label_encoder.transform([cls])[0]: count for cls, count in image_counts.items()}
plt.figure(figsize=(16, 6))
sns.barplot(x= list(encoded_counts.keys()), y= list(encoded_counts.values()), palette="viridis")

plt.xlabel("Class (Encoded)", fontsize=14)
plt.ylabel("Number of Images", fontsize=14)
plt.title(f"Dataset Distribution ({num_classes} Classes)", fontsize=16)
plt.xticks([])
plt.show()

def show_sample_images(num_samples=5):
    fig, axes = plt.subplots(num_samples, num_samples, figsize=(10, 10))
    axes = axes.ravel()

    for i, class_name in enumerate(np.random.choice(class_names, num_samples**2, replace=False)):
        img_folder = os.path.join(DATA_DIR, class_name)
        img_file = np.random.choice(os.listdir(img_folder))
        img_path = os.path.join(img_folder, img_file)

        # Đọc ảnh
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Hiển thị
        axes[i].imshow(img)
        axes[i].set_title(class_name, fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

show_sample_images()


DATA_DIR = r"D:\datn_haui\new_model\dataset\fruits-360\Training"

image_sizes = []

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)

    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                h, w, c = img.shape
                image_sizes.append((w, h))
size_counts = Counter(image_sizes)

for size, count in size_counts.items():
    print(f"Kích thước {size}: {count} ảnh")



