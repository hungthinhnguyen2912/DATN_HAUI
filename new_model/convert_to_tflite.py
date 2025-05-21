import tensorflow as tf
import keras
model = keras.models.load_model("D:/datn_haui/new_model/keras/model_checkpoint_epoch_10.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("output/360.tflite", "wb") as f:
    f.write(tflite_model)

print("Convert complete")
