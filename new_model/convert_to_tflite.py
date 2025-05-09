import tensorflow as tf
import keras

model = keras.models.load_model("model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model_4.tflite", "wb") as f:
    f.write(tflite_model)

print("Convert complete")
