import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import numpy as np

# Đường dẫn TFRecord và dataset gốc
TFRECORD_FILE = "tfrecodrd_file\\train_data.tfrecord"
DATA_DIR = "D:\\datn_haui\\new_model\\dataset\\fruits-360\\Training"

# Đọc số lượng mẫu
count = sum(1 for _ in tf.data.TFRecordDataset(TFRECORD_FILE))
print(f"📊 TFRecord chứa {count} mẫu.")

# Hàm parse TFRecord
def parse_example(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = example['label']
    return image, label

# Load TFRecord dataset và shuffle
dataset = tf.data.TFRecordDataset(TFRECORD_FILE)
dataset = dataset.map(parse_example).shuffle(3000)

# Chọn 5 ảnh ngẫu nhiên
samples = list(dataset.take(5))

# Tạo danh sách để tìm ảnh gốc
selected_labels = [label.numpy() for _, label in samples]

# Duyệt qua thư mục dataset để tìm ảnh tương ứng
def find_original_images(selected_labels, data_dir):
    original_images = []
    class_dirs = sorted(os.listdir(data_dir))  # Sắp xếp class theo thứ tự label
    for label in selected_labels:
        class_name = class_dirs[label]  # Tìm thư mục theo index label
        class_path = os.path.join(data_dir, class_name)
        img_files = os.listdir(class_path)
        random_img = random.choice(img_files)  # Chọn 1 ảnh bất kỳ trong class đó
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
