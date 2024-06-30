# SECTION 1 - SETUP ####

### numpy 1.19.5
### scipy 1.10
### tensorflow 2.5

import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import scipy

# SECTION 2 Convolution/Pooling/Flattening/Dense

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# SECTION 3 Fitting images with CNN

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    "C:/Users/hasan/OneDrive/Masaüstü/opencv-exercise/pythonProject1/dataset/training_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    "C:/Users/hasan/OneDrive/Masaüstü/opencv-exercise/pythonProject1/dataset/test_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

model.fit(
    training_set,
    steps_per_epoch=80,
    epochs=10,
    validation_data=test_set,
    validation_steps=46
)

# SECTION 4 The Classification with CNN

def predict_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    return "plane" if result[0][0] == 1 else "non-plane"


def draw_rectangle(type):
    if type == "plane":
        pass
    else :
        pass

print(predict_image("C:/Users/hasan/OneDrive/Masaüstü/opencv-exercise/pythonProject1/dataset/ucak.jpg"))
print(predict_image("C:/Users/hasan/OneDrive/Masaüstü/opencv-exercise/pythonProject1/dataset/kopek.jpg"))
print(predict_image("C:/Users/hasan/OneDrive/Masaüstü/opencv-exercise/pythonProject1/dataset/ucak2.jpg"))
print(predict_image("C:/Users/hasan/OneDrive/Masaüstü/opencv-exercise/pythonProject1/dataset/kedı.jpg"))

