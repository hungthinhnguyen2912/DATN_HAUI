import os
import keras

img_height = 224
img_width = 224
batch_size = 64

train_data_dir = r'D:\datn_haui\mobilenetv2\dataset\fruits-360\Training'
print(train_data_dir)
train_ds = keras.utils.image_dataset_from_directory(
    train_data_dir,
    label_mode='categorical',
    image_size = (img_height, img_width),
    batch_size = batch_size
)
print(train_ds)

val_data_dir = r'D:\datn_haui\mobilenetv2\dataset\fruits-360\Test'
print(val_data_dir)
val_ds = keras.utils.image_dataset_from_directory(
    val_data_dir,
    label_mode='categorical',
    image_size = (img_height, img_width),
    batch_size = batch_size
)
print(val_ds)

train_classes = os.listdir("D:/datn_haui/mobilenetv2/dataset/fruits-360/Training")
test_classes = os.listdir("D:/datn_haui/mobilenetv2/dataset/fruits-360/Test")
print(train_classes)
print(test_classes)
