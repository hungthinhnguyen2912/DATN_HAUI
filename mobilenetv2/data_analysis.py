import os
import tensorflow as tf
import keras
from keras.src.layers import Rescaling

train_data_dir = r'D:\datn_haui\mobilenetv2\dataset\fruits-360\Training'
val_data_dir = r'D:\datn_haui\mobilenetv2\dataset\fruits-360\Test'
train_classes = os.listdir("D:/datn_haui/mobilenetv2/dataset/fruits-360/Training")
test_classes = os.listdir("D:/datn_haui/mobilenetv2/dataset/fruits-360/Test")

# Numbers of class: 170
print(train_classes)
print(test_classes)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

image_count = 0
for root, dirs, files in os.walk(train_data_dir):
    image_count += sum(f.lower().endswith(image_extensions) for f in files)

for root, dirs, files in os.walk(val_data_dir):
    image_count += sum(f.lower().endswith(image_extensions) for f in files)

print(f"Total image: {image_count}")
# Total image: 114226
# Numbers images of each classes
class_counts = {class_name: len(os.listdir(os.path.join(train_data_dir, class_name)))
                for class_name in os.listdir(train_data_dir)}

class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1]))

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
print("-------------------")
class_counts = {class_name: len(os.listdir(os.path.join(val_data_dir, class_name)))
                for class_name in os.listdir(val_data_dir)}

class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1]))

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")



# train_data_set = keras.utils.image_dataset_from_directory(
#     train_data_dir,
#     image_size= (224, 224),
#     batch_size= 64,
#     shuffle= True,
#     label_mode= "categorical"
# )
#
# val_data_set = keras.utils.image_dataset_from_directory(
#     val_data_dir,
#     image_size= (224, 224),
#     batch_size= 32,
#     shuffle= True,
#     label_mode= "categorical"
# )
#
# normalization_layer = Rescaling(1./255)
# train_data_set = train_data_set.map(lambda x, y: (normalization_layer(x), y))
# val_data_set = val_data_set.map(lambda x, y: (normalization_layer(x), y))
#
# print(train_data_set)
