import tensorflow as tf
import matplotlib.pyplot as plt

path = r"tfrecodrd_file\\train_data.tfrecord"


# Hàm parse tfrecord (giả sử bạn lưu ảnh dưới dạng raw bytes)
def _parse_function(proto):
    # Tùy thuộc vào cấu trúc bạn đã dùng để tạo TFRecord
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        # thêm label nếu cần
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    image = tf.io.decode_jpeg(parsed_features['image'], channels=3)  # hoặc decode_png
    return image


# Đọc file TFRecord
raw_dataset = tf.data.TFRecordDataset(path)
parsed_dataset = raw_dataset.map(_parse_function)

# Lấy 1 ảnh mẫu ra
for image in parsed_dataset.take(1):
    img_tensor = image.numpy()
    break
# In giá trị pixel đầu tiên
print("Pixel đầu tiên (H x W x C):", img_tensor[0, 0])
# Bản gốc (decode trực tiếp)
plt.subplot(1, 2, 1)
plt.imshow(img_tensor)
plt.title("Ảnh gốc decode")

# Đảo kênh (BGR → RGB)
img_rgb = img_tensor[..., ::-1]
plt.subplot(1, 2, 2)
plt.imshow(img_rgb)
plt.title("Đảo màu (BGR → RGB)")

plt.show()

# Hoặc xem thử ảnh bằng matplotlib
plt.imshow(img_tensor)
plt.title("Check Color Order")
plt.show()
