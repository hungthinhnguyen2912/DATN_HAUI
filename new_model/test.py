import os
import tensorflow as tf
import keras
import pandas as pd
from keras import Input, Model
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from keras.src.regularizers import L2

# Dataset path
base_dir = r"D:\\datn_haui\\new_model\\dataset\\Vegetable Images"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "validation")
val_dir = os.path.join(base_dir, "test")

# Configs
image_size = 224
batch_size = 32
num_classes = 15

def prepare_data():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def build_model(dropout_rate=0.1, unfreeze_layers=2):
    inputs = Input(shape=(image_size, image_size, 3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)

    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=L2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)  # Dropout sẽ bị vô hiệu khi convert TFLite
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

def train_model(model, train_gen, val_gen):
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC(name='auroc')]
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(patience=2, factor=0.2, min_lr=1e-6)
        ]
    )

    pd.DataFrame(history.history).to_csv("training_log.csv", index=False)
    return model

def evaluate_and_export(model, test_gen):
    loss, acc, prec, rec, auc = model.evaluate(test_gen)
    print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")

    # Save full model
    model.save("vegetable_model.keras")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("vegetable_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Model exported to vegetable_model.tflite")

if __name__ == '__main__':
    train_gen, val_gen, test_gen = prepare_data()
    model = build_model(dropout_rate=0.1, unfreeze_layers=2)
    model = train_model(model, train_gen, val_gen)
    evaluate_and_export(model, test_gen)