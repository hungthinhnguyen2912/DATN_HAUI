import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import numpy as np

TFRECORD_FILE = "tfrecodrd_file\\train_data.tfrecord"
DATA_DIR = "D:\\datn_haui\\new_model\\dataset\\fruits-360\\Training"

count = sum(1 for _ in tf.data.TFRecordDataset(TFRECORD_FILE))
print(f"📊 TFRecord chứa {count} mẫu.")

def parse_example(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.reverse(image, axis=[-1])  # Chuyển từ BGR sang RGB
    label = example['label']
    return image, label

# Đọc dataset
dataset = tf.data.TFRecordDataset(TFRECORD_FILE)
dataset = dataset.map(parse_example)

label_counts = {}
for _, label in dataset:
    label = label.numpy()
    label_counts[label] = label_counts.get(label, 0) + 1

labels = sorted(label_counts.keys())
counts = [label_counts[label] for label in labels]
num_classes = len(labels)

plt.figure(figsize=(12, 6))
hist, bins, patches = plt.hist(labels, bins=num_classes, weights=counts, edgecolor='black')

colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
for patch, color in zip(patches, colors):
    patch.set_facecolor(color)

plt.title(f"Dataset Distribution ({num_classes} Classes)")
plt.xlabel("Class (Encoded)")
plt.ylabel("Number of Images")
plt.show()

dataset = dataset.shuffle(buffer_size=count, reshuffle_each_iteration=True)
samples = list(dataset.take(5))
selected_labels = [label.numpy() for _, label in samples]

# Tìm ảnh gốc
def find_original_images(selected_labels, data_dir):
    original_images = []
    class_dirs = sorted(os.listdir(data_dir))
    for label in selected_labels:
        class_name = class_dirs[label]
        class_path = os.path.join(data_dir, class_name)
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            print(f"Warning: No valid images found in {class_path}")
            continue
        random_img = random.choice(img_files)
        img_path = os.path.join(class_path, random_img)
        original_images.append((Image.open(img_path), label))
    return original_images

# Lấy ảnh gốc
original_images = find_original_images(selected_labels, DATA_DIR)

# Hiển thị ảnh từ TFRecord
plt.figure(figsize=(10, 5))
for i, (image, label) in enumerate(samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image.numpy())
    plt.axis("off")
    plt.title(f"TFRecord Label: {label.numpy()}")

# Hiển thị ảnh gốc dưới ảnh từ TFRecord
for i, (image, label) in enumerate(original_images):
    plt.subplot(2, 5, i + 6)
    plt.imshow(np.array(image))
    plt.axis("off")
    plt.title(f"Original Label: {label}")

plt.show()