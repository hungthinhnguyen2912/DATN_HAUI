import tensorflow as tf
import matplotlib.pyplot as plt

import keras

TFRECORD_FILE = "tfrecodrd_file\\train_data.tfrecord"

# Đếm tổng số mẫu
count = sum(1 for _ in tf.data.TFRecordDataset(TFRECORD_FILE))
print(f"📊 TFRecord chứa {count} mẫu.")

def parse_and_preprocess(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reverse(image, axis=[-1])
    image = keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
    label = example['label']
    return image, label


# Đọc dataset
dataset = tf.data.TFRecordDataset(TFRECORD_FILE)
dataset = dataset.map(parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

train_size = int(0.8 * count)
val_size = count - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

BATCH_SIZE = 32
steps_per_epoch = train_size // BATCH_SIZE
validation_steps = val_size // BATCH_SIZE

train_dataset = train_dataset.shuffle(buffer_size=min(train_size, 10000)).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for image, label in val_dataset.take(1):
    print(image.shape, label.numpy())