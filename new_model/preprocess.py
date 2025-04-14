import tensorflow as tf
import os
import cv2
import random
import numpy as np
from tqdm import tqdm

DATASET_DIR = r"D:\\datn_haui\\new_model\\dataset\\fruits-360\\Training"
TFRECORD_FILE = r"tfrecodrd_file\\train_data.tfrecord"

# Kiểm tra thư mục
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory {DATASET_DIR} does not exist!")
os.makedirs(os.path.dirname(TFRECORD_FILE), exist_ok=True)

classes = sorted(os.listdir(DATASET_DIR))
class_to_index = {cls: idx for idx, cls in enumerate(classes)}
TARGET_COUNT = 300

def augment_image(image, img_path=None):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-10, 10)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if random.random() > 0.3:
        noise = np.random.normal(0, 3, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
    image = np.clip(image, 0, 255)
    if (np.mean(image) < 20 or np.std(image) < 10 or np.max(image) < 50 or np.min(image) > 200):
        print(f"Warning: Augmented image {img_path} is corrupted, using original image")
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"Error: Could not read original image {img_path}")
            return None
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        return original_image
    return image

def serialize_example(image, label, filename):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def process_image(img_path, augment=False):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read image {img_path}")
        return None
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if augment:
        image = augment_image(image, img_path)
        if image is None:
            return None
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, image_encoded = cv2.imencode('.jpg', image)
    return image_encoded.tobytes()

with tf.io.TFRecordWriter(TFRECORD_FILE) as writer:
    for cls in tqdm(classes, desc="Processing classes"):
        cls_path = os.path.join(DATASET_DIR, cls)
        label = class_to_index[cls]
        image_paths = [os.path.join(cls_path, entry.name) for entry in os.scandir(cls_path) 
                       if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        original_count = len(image_paths)

        if original_count >= TARGET_COUNT:
            image_paths = random.sample(image_paths, TARGET_COUNT)

        for img_path in tqdm(image_paths, desc=f"Writing {cls}", leave=False):
            image_bytes = process_image(img_path)
            if image_bytes:
                writer.write(serialize_example(image_bytes, label, os.path.basename(img_path)))

        if original_count < TARGET_COUNT:
            needed = TARGET_COUNT - original_count
            for _ in tqdm(range(needed), desc=f"Augmenting {cls}", leave=False):
                img_path = random.choice(image_paths)
                image_bytes = process_image(img_path, augment=True)
                if image_bytes:
                    writer.write(serialize_example(image_bytes, label, os.path.basename(img_path)))

if os.path.exists(TFRECORD_FILE) and os.path.getsize(TFRECORD_FILE) > 0:
    print(f"✅ TFRecord file created successfully at: {TFRECORD_FILE}")
else:
    print(f"❌ Failed to create TFRecord file at: {TFRECORD_FILE}")