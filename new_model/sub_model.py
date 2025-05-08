import keras
import numpy as np
from keras import Model
from sklearn.metrics.pairwise import cosine_similarity
from train import train_generator, test_labels

full_model = keras.models.load_model('model.keras')
feature_model = Model(inputs=full_model.input, outputs=full_model.get_layer(index=-2).output)
features = []
labels = []

for i in range(len(train_generator)):
    x_batch, y_batch = train_generator[i]
    feature_batch = feature_model.predict(x_batch)
    features.append(feature_batch)
    labels.append(np.argmax(y_batch, axis=1))

features = np.vstack(features)
labels = np.hstack(labels)

prototypes = {}
for label in np.unique(labels):
    class_features = features[labels == label]
    prototypes[label] = np.mean(class_features, axis=0)

def predict_with_unknown(image):
    feature = feature_model.predict(np.expand_dims(image, axis=0))[0]  # shape: (64,)
    sims = {label: cosine_similarity([feature], [proto])[0][0] for label, proto in prototypes.items()}
    max_label, max_score = max(sims.items(), key=lambda x: x[1])

    threshold = 0.75
    if max_score < threshold:
        return 'unknown'
    else:
        return list(test_labels)[max_label]

import tensorflow as tf

feature_model = Model(inputs=full_model.input, outputs=full_model.get_layer(index=-2).output)

converter = tf.lite.TFLiteConverter.from_keras_model(feature_model)
tflite_model = converter.convert()

with open("feature_model.tflite", "wb") as f:
    f.write(tflite_model)