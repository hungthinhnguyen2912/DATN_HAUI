import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt

dataset = r"D:\\datn_haui\\new_model\\dataset\\Vegetable Images"

train_dir = os.path.join(dataset, "train")
test_dir = os.path.join(dataset, "validation")
val_dir = os.path.join(dataset, "test")

# Initialize list to store dimensions (height, width, channels)
image_dimensions = []

# Get list of classes
classes = os.listdir(train_dir)

plt.figure(figsize=(15, 10))

# Loop through each class and process the images
for i, class_name in enumerate(classes):
    class_dir = os.path.join(train_dir, class_name)
    images = os.listdir(class_dir)

    # Read the first image in the class directory
    image_path = os.path.join(class_dir, images[0])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save height, width, and number of channels
    height, width, channels = image.shape
    image_dimensions.append((height, width, channels))

    # Plot the image
    plt.subplot(3, 5, i+1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis('off')

plt.show()

df = pd.DataFrame(image_dimensions, columns=["Height", "Width", "Channels"])
var = df.describe().loc[["min", "mean", "max"]]
print(var)

count_dict = {}

for root, dirs, files in os.walk(train_dir):
    clase = os.path.basename(root)
    count_dict[clase] = len(files)

lables = list(count_dict.keys())[1:]
count = list(count_dict.values())[1:]
plt.figure(figsize=(20,4))
plt.bar(lables, count, color="skyblue")
plt.title("Class distribution")
plt.xlabel("Vegetable")
plt.ylabel("Quantity")
plt.show()