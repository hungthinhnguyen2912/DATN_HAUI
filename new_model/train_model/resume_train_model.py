import keras
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.src.optimizers import Adam
from keras.src.regularizers import L2
from keras import Input, Model
from keras.src.callbacks import ModelCheckpoint
from train_model.fruit360 import train_generator, val_generator, reduce_lr, early_stopping
import os

checkpoint_path = "/keras/model_at_epoch_6.keras"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Không tìm thấy file tại: {checkpoint_path}")

initial_epoch = 6

def load_model(n_layers_to_unfreeze=0):
    inputs = Input(shape=(224, 224, 3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs, input_shape=(224, 224, 3))
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
    target_output = Dense(199, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=target_output)
    return model

def resume_train():
    model = load_model()
    model.load_weights(checkpoint_path)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC(name='auroc')])

    checkpoint_cb = ModelCheckpoint(
        filepath="output/model_checkpoint_epoch_{epoch:02d}.keras",
        save_best_only=False,
        monitor='val_loss',
        save_weights_only=False,
        verbose=1
    )

    history = model.fit(
        train_generator,
        epochs=10,
        initial_epoch=initial_epoch,
        validation_data=val_generator,
        callbacks=[early_stopping, reduce_lr, checkpoint_cb]
    )
    model.save('D:/datn_haui/new_model/model_360.keras')
    # Lưu history để tránh mất
    import pandas as pd
    os.makedirs("output", exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("D:/datn_haui/new_modeloutput/history_epochs_7_to_10.csv", index=False)
    print("Lịch sử huấn luyện đã được lưu tại:D:/datn_haui/new_model output/history_epochs_7_to_10.csv")

resume_train()