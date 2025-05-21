import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
from keras.src.regularizers import L2
from keras import Input, Model
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam

dataset = r"D:\\datn_haui\\new_model\\dataset\\Vegetable Images"

train_dir = os.path.join(dataset, "train")
test_dir = os.path.join(dataset, "validation")
val_dir = os.path.join(dataset, "test")

image_size = 224
batch_size = 32

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
    class_mode='categorical',
    color_mode='rgb'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)


def load_model(n_layers_to_unfreeze=0):
    inputs = Input(shape=(224, 224, 3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    n_layers_to_unfreeze = min(n_layers_to_unfreeze, len(base_model.layers))
    for layer in base_model.layers:
        layer.trainable = False
    if n_layers_to_unfreeze > 0:
        for layer in base_model.layers[-n_layers_to_unfreeze:]:
            layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(64, kernel_regularizer=L2(0.0001), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    target_output = Dense(15, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=target_output)

    return model


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-6
)


def plot(history):
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


def train_model(model,
                train_generator,
                val_generator,
                epochs,
                learning_rate,
                callbacks=None):
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=[
                      'accuracy',
                      keras.metrics.Precision(),
                      keras.metrics.Recall(),
                      keras.metrics.AUC(name='auroc')
                  ])

    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=val_generator,
                        callbacks=callbacks)
    history_df = pd.DataFrame(history.history)
    model.save("model_vegetable_image.keras")
    plot(history_df)


model = load_model(10)

train_model(model,
            train_generator,
            val_generator,
            epochs=10,
            learning_rate=0.001,
            callbacks=[early_stopping, reduce_lr])

true_classes = test_generator.classes
predictions = model.predict(test_generator)

predicted_classes = np.argmax(predictions, axis=1)

test_labels = test_generator.class_indices.keys()

model.evaluate(test_generator)