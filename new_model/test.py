import datetime
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import gdown
import os

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import class_weight

policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

# Tải file TFRecord từ Google Drive
file_id = "1UPXZao-dV6SAeEfhUSpSOumVva6Tt1fo"
output = "train_data.tfrecord"
if not os.path.exists(output):
    print("📥 Đang tải TFRecord từ Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
else:
    print("✅ TFRecord đã tồn tại, bỏ qua bước tải.")

TFRECORD_FILE = output

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
count = 170 * 300
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# Đếm số mẫu
# Parse và preprocess
def parse_and_preprocess(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
    label = example['label']
    return image, label


raw_dataset = tf.data.TFRecordDataset(TFRECORD_FILE)
dataset = raw_dataset.map(parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

labels = [label.numpy() for _, label in dataset]
num_classes = len(np.unique(labels))
print(f"Số lớp: {num_classes}")

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

shuffled_dataset = dataset.shuffle(buffer_size=2000, reshuffle_each_iteration=False)
train_size = int(0.8 * count)
val_size = count - train_size
train_dataset = shuffled_dataset.take(train_size)
val_dataset = shuffled_dataset.skip(train_size)

train_dataset = train_dataset.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load MobileNetV2
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
NUM_CLASSES = 170
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')
])
INIT_LR = 0.001
EPOCHS = 100

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=EPOCHS,
    decay_rate=INIT_LR / EPOCHS,
    staircase=True
)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]
# Tính steps_per_epoch
steps_per_epoch = (train_size + BATCH_SIZE - 1) // BATCH_SIZE
validation_steps = (val_size + BATCH_SIZE - 1) // BATCH_SIZE
base_model.trainable = True

fine_tune_at = 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS_PHASE2 = 100
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE2,
    steps_per_epoch=math.ceil(train_size / EPOCHS_PHASE2),
    validation_steps=math.ceil(val_size / EPOCHS_PHASE2),
    callbacks=callbacks,
)

# Đánh giá trên tập validation
val_loss, val_accuracy = model.evaluate(val_dataset, steps=validation_steps)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Đánh giá chi tiết: precision, recall, F1-score
y_true = []
y_pred = []
for images, labels in val_dataset:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1-score: {f1:.4f}")

# Vẽ đồ thị
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Lưu mô hình
model.save('tl_mobilenetv2_fruit_classifier.keras')
