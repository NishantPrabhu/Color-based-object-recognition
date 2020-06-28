
# Dependencies
import os
import numpy as np
from glob import glob
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical


# Generate dataset from images present in disk
# One image belongs to one class

x_train, y_train = [], []

folders = os.listdir('../../data')
folders.remove('test_images')
progress = tqdm(total=len(folders), position=0, desc='Progress', leave=False)
folder_name = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
label = 0

for fol in folders:
    folder_name.set_description_str(f'Now processing {fol} ...')
    paths = glob('../../data/'+fol+'/*.jpg')
    folder_pbar = tqdm(total=len(paths), position=2, desc='Folder', leave=False)
    for path in paths:
        img = load_img(path, target_size=(299, 299))
        img = img_to_array(img)
        img = preprocess_input(img)

        x_train.append(img)
        y_train.append(label)
        folder_pbar.update(1)
        label += 1

    progress.update(1)

x_train, y_train = np.array(x_train), np.array(y_train)
y_train = to_categorical(y_train)


# Train resnet's output layer for image classification
resnet_model = ResNet50(weights='imagenet')
resnet_model = Model(resnet_model.inputs, resnet_model.layers[-2].output)
for layer in resnet_model.layers:
    layer.trainable = False

# Create main model
model = Sequential([
    resnet_model,
    Dense(len(y_train), activation='softmax', use_bias=True)
])
model.compile(loss='categorical_crossentropy', optimizer='adam')


# Create ImageDataGenerator object
image_gen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=50,
    width_shift_range=0.5,
    height_shift_range=0.5,
    brightness_range=[-1.0, 1.0],
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True
)

image_gen.fit(x_train)

# Fit the model to the generator
model.fit_generator(image_gen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) // 32,
                    epochs = 10)


# Save the trained model
model.save('../../saved_data/23 Jun/resnet_classifier.h5')
