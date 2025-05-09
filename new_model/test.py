from keras import Input
inputs = Input(shape=(224, 224, 3))


from keras.src.applications.mobilenet_v2 import MobileNetV2

base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)