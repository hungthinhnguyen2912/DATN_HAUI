import tensorflow as tf
import keras

# Load mô hình từ file .keras
model = keras.models.load_model("model.keras")

# Convert sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Lưu file .tflite
with open("model_4.tflite", "wb") as f:
    f.write(tflite_model)

print("Convert complete")
