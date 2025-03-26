import cv2
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_DIR = r"D:\\datn_haui\\mobilenetv2\\dataset\\fruits-360\\Training"
TFRECORD_FILE = r"tfrecord_dataset\\train_3.tfrecord"
IMG_SIZE = (224, 224)
AUGMENT_SIZE = 150
SELECT_SIZE = 300


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image


def process_and_save_tfrecord():
    writer = tf.io.TFRecordWriter(TFRECORD_FILE)

    total_images = sum(len(os.listdir(os.path.join(DATA_DIR, class_name))) for class_name in os.listdir(DATA_DIR))
    with tqdm(total=total_images, desc="Processing Images", unit="img", colour="green", position=0, leave=True) as pbar:
        for label, class_name in enumerate(os.listdir(DATA_DIR)):
            class_path = os.path.join(DATA_DIR, class_name)
            images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
            if len(images) > 600:
                print(f'Class {class_name} có {len(images)} ảnh, chọn ngẫu nhiên 300 ảnh')
                images = random.sample(images, SELECT_SIZE)

            elif len(images) < 100:
                print(f"Class {class_name} có {len(images)} ảnh, augment thêm 100 ảnh")
                augmented_images = []
                while len(augmented_images) < AUGMENT_SIZE:
                    img_path = random.choice(images)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, IMG_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                    img = augment_image(img).numpy()
                    augmented_images.append(img)
                images.extend(augmented_images)

            else:
                print(f"Class {class_name} có {len(images)} ảnh, giữ nguyên không thay đổi.")
            for img_path in images:
                if isinstance(img_path, str):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, IMG_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = img_path * 255.0
                    img = img.astype(np.uint8)

                feature = {
                    "image": _bytes_feature(tf.io.encode_jpeg(img).numpy()),
                    "label": _int64_feature(label)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

                del img
                pbar.update(1)

        pbar.close()

    writer.close()
    print(f"✅ TFRecord saved: {TFRECORD_FILE}")


if __name__ == "__main__":
    process_and_save_tfrecord()
