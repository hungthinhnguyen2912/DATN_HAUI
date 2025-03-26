import os
from PIL import Image


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

image_shapes = []
for class_name in train_classes:
    img_path = os.path.join(train_data_dir, class_name, os.listdir(os.path.join(train_data_dir, class_name))[0])
    img = Image.open(img_path)
    image_shapes.append(img.size)

# The number of size, and it's popularity
from collections import Counter
print("The image size most popular", Counter(image_shapes).most_common(5))
# The image size most popular [((100, 100), 161), ((224, 224), 5), ((1000, 752), 1), ((1501, 1500), 1), ((612, 410), 1)]

