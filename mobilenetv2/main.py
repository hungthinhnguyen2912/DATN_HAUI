import datetime
import os
import tensorflow as tf
import keras.backend as K

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
AUTOTUNE = tf.data.experimental.AUTOTUNE

DATA_DIR = r"D:\\datn_haui\\mobilenetv2\\dataset\\fruits-360\\Training"
class_num = len([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

TFRECORD_TRAIN = "tfrecord_dataset/train_3.tfrecord"
TFRECORD_VAL = "tfrecord_dataset/test_3.tfrecord"


def _parse_function(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed_example["image"], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Đọc nhãn (one-hot encoding)
    label = tf.one_hot(parsed_example["label"], depth=class_num)

    return image, label


def load_tfrecord_dataset(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    dataset = dataset.map(_parse_function, num_parallel_calls=1)
    dataset = dataset.shuffle(500).batch(BATCH_SIZE).prefetch(1)
    return dataset


train_dataset = load_tfrecord_dataset(TFRECORD_TRAIN)
val_dataset = load_tfrecord_dataset(TFRECORD_VAL)

# Load MobileNetV2
model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    input_shape=(224, 224, 3),
    include_top=False
)
model.trainable = False

Top_layers = tf.keras.layers.GlobalAveragePooling2D()(model.output)
Top_layers = tf.keras.layers.Dense(512, activation='relu')(Top_layers)  #
Top_layers = tf.keras.layers.Dropout(0.25)(Top_layers)
Top_layers = tf.keras.layers.Dense(class_num, activation='softmax')(Top_layers)

MobileNet_Model = tf.keras.Model(inputs=model.input, outputs=Top_layers)
MobileNet_Model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

MobileNet_Model.summary()
logdir = os.path.join("logs", datetime.datetime.now().strftime("MobileNet_Model-%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

for epoch in range(10):
    print(f"🔄 Epoch {epoch + 1}/10")
    MobileNet_Model.fit(train_dataset, validation_data=val_dataset, epochs=1, callbacks=[tensorboard_callback])
    K.clear_session()
MobileNet_Model.save('saved_models/MobileNetV2_Model.keras')
