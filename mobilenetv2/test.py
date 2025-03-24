import os
import tensorflow as tf
import numpy as np

train_data_dir = r"D:\datn_haui\mobilenetv2\dataset\fruits-360\Training"
min_images = 100
num_augmented_images = 200
valid_extensions = (".jpg", ".jpeg", ".png")
class_counts = {
    class_name: len([
        f for f in os.listdir(os.path.join(train_data_dir, class_name))
        if f.lower().endswith(valid_extensions)
    ])
    for class_name in os.listdir(train_data_dir)
}
class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1]))
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return img

def augment_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.rot90(img, k=np.random.randint(4))
    return img

for class_name, count in class_counts.items():
    if count >= min_images:
        continue

    input_dir = os.path.join(train_data_dir, class_name)
    output_dir = os.path.join(train_data_dir, class_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📌 Class '{class_name}' có {count} ảnh, đang augment thêm {num_augmented_images} ảnh...")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    for i in range(num_augmented_images):
        img_name = np.random.choice(image_files)
        img_path = os.path.join(input_dir, img_name)
        img = load_image(img_path)
        aug_img = augment_image(img)
        aug_img = tf.image.encode_jpeg(tf.cast(aug_img, tf.uint8))
        base_name, ext = os.path.splitext(img_name)
        new_file = os.path.join(output_dir, f"{base_name}_aug_{i}.jpg")
        tf.io.write_file(new_file, aug_img)
    print(f"✅ Đã augment xong class '{class_name}'! Tổng số ảnh mới: {count + num_augmented_images}")
print("🎉 Hoàn thành!")
