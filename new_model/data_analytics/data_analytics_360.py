import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# Đường dẫn tới Fruit 360 dataset
dataset = r"D:\datn_haui\new_model\dataset\fruits-360_100x100\fruits-360"

# Thư mục train/test/validation
train_dir = os.path.join(dataset, "Training")
test_dir = os.path.join(dataset, "Test")

# Danh sách lưu kích thước ảnh (cao, rộng, kênh)
image_dimensions = []

# Lấy danh sách các class
classes = os.listdir(train_dir)

# Vẽ ảnh đại diện của từng class
plt.figure(figsize=(15, 10))

for i, class_name in enumerate(classes[:15]):  # Vẽ 15 class đầu
    class_dir = os.path.join(train_dir, class_name)
    images = os.listdir(class_dir)

    if not images:
        continue

    image_path = os.path.join(class_dir, images[0])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, channels = image.shape
    image_dimensions.append((height, width, channels))

    plt.subplot(3, 5, i + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Phân tích thống kê hình ảnh
df = pd.DataFrame(image_dimensions, columns=["Height", "Width", "Channels"])
var = df.describe().loc[["min", "mean", "max"]]
print(var)

# Thống kê số lượng ảnh mỗi class
target_classes = os.listdir(train_dir)
training_set_distribution = [len(os.listdir(os.path.join(train_dir, DIR))) for DIR in os.listdir(train_dir)]
testing_set_distribution = [len(os.listdir(os.path.join(test_dir, DIR))) for DIR in os.listdir(test_dir)]
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(30, 6))

ind = np.arange(len(target_classes))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = ax.bar(ind, training_set_distribution, width)
p2 = ax.bar(ind, testing_set_distribution, width, bottom=training_set_distribution)

plt.ylabel('Images')
plt.title('Distribution', fontsize=18)
plt.xticks(ind, target_classes, rotation=90)
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Training', 'Testing'))
plt.show()