
# Dependencies
import os
from tqdm import tqdm
import numpy as np
from glob import glob
from ntpath import basename
from warnings import filterwarnings
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

filterwarnings(action='ignore')


# Arguments

N_SAMPLES = 100
img_paths = glob('../../data/Biscuits/*.png')

progress_bar = tqdm(total=len(img_paths), position=0, desc='Progress')
current_file = tqdm(total=0, position=1, bar_format='{desc}')

print('')

for path in img_paths:

    name = basename(path).split('.')[0]
    current_file.set_description_str(f'Now processing {name} ...')
    os.mkdir('../../image_data/' + name)
    save_path = '../../image_data/' + name

    image = load_img(path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    aug = ImageDataGenerator(
        rotation_range=40,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Number of generated images counter
    total = 0

    # Image generator object
    imageGen = aug.flow(image, batch_size=1, save_to_dir=save_path,
                        save_prefix='name', save_format='png')

    for image in imageGen:
        total += 1
        if total == N_SAMPLES:
            break

    progress_bar.update(1)
