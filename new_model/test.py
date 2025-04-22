import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import callbacks
import math
import time
import gdown

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

# Đếm số mẫu
count = sum(1 for _ in tf.data.TFRecordDataset(TFRECORD_FILE))
print(f"📊 TFRecord chứa {count} mẫu.")

# Real-time augmentation
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.05)
    return image, label

# Parse và preprocess
def parse_and_preprocess(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
    label = example['label']
    return image, label

# Tạo dataset
raw_dataset = tf.data.TFRecordDataset(TFRECORD_FILE)
dataset = raw_dataset.map(parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle và chia dataset
shuffled_dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=False)
train_size = int(0.8 * count)
val_size = count - train_size
train_dataset = shuffled_dataset.take(train_size)
val_dataset = shuffled_dataset.skip(train_size)

# Batch size tăng lên 64
BATCH_SIZE = 64
steps_per_epoch = math.ceil(train_size / BATCH_SIZE)
validation_steps = math.ceil(val_size / BATCH_SIZE)

# Thêm real-time augmentation cho tập train
train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache().shuffle(buffer_size=10000, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load MobileNetV2
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False
NUM_CLASSES = 170

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')
])

# Compile mô hình
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_model_phase1.keras', monitor='val_loss', save_best_only=True),
    keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

total_start_time = time.time()

# Phase 1: Freeze base model
EPOCHS_PHASE1 = 10
history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)

print('Phase 2: Start fine-tune model')

# Phase 2: Fine-tune từ layer 80
base_model.trainable = True
fine_tune_at = 80
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Dùng lại Adam cho phase 2
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS_PHASE2 = 10
history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# Gộp history 2 phase
history = {
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
}

# save model
model.save('mobilenetv2_finetuned_final.keras')
total_end_time = time.time()
print(f"Tổng thời gian huấn luyện: {(total_end_time - total_start_time) / 60:.2f} phút")

# Vẽ biểu đồ
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.close()

# Đánh giá
val_loss, val_accuracy = model.evaluate(val_dataset, steps=validation_steps)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Thời gian xử lý: {time.process_time()} giây")
