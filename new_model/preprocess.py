import tensorflow as tf
import os
import cv2
import random
import numpy as np
from tqdm import tqdm

DATASET_DIR = r"D:\\datn_haui\\new_model\\dataset\\fruits-360\\Training"
TFRECORD_FILE = r"tfrecodrd_file\\train_data.tfrecord"

classes = sorted(os.listdir(DATASET_DIR))
class_to_index = {cls: idx for idx, cls in enumerate(classes)}
TARGET_COUNT = 300


def augment_image(image):
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random rotation
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))

    # Random brightness/contrast
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-30, 30)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Add noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)

    return image


def serialize_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def process_image(img_path, augment=False):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if augment:
        image = augment_image(image)

    _, image_encoded = cv2.imencode('.jpg', image)
    return image_encoded.tobytes()


with tf.io.TFRecordWriter(TFRECORD_FILE) as writer:
    for cls in tqdm(classes, desc="Processing classes"):
        cls_path = os.path.join(DATASET_DIR, cls)
        label = class_to_index[cls]

        # Get all image paths
        image_paths = [os.path.join(cls_path, img) for img in os.listdir(cls_path)]
        original_count = len(image_paths)

        # If we have more than needed, sample randomly
        if original_count >= TARGET_COUNT:
            image_paths = random.sample(image_paths, TARGET_COUNT)

        # Write original images
        for img_path in tqdm(image_paths, desc=f"Writing {cls}", leave=False):
            image_bytes = process_image(img_path)
            writer.write(serialize_example(image_bytes, label))

        # If we need more images, generate augmented ones
        if original_count < TARGET_COUNT:
            needed = TARGET_COUNT - original_count
            for _ in tqdm(range(needed), desc=f"Augmenting {cls}", leave=False):
                img_path = random.choice(image_paths)
                image_bytes = process_image(img_path, augment=True)
                writer.write(serialize_example(image_bytes, label))

print(f"✅ TFRecord file created successfully at: {TFRECORD_FILE}")