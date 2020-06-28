
# Dependencies
import os
from glob import glob
from ntpath import basename
from shutil import copyfile


root_folder = '../../image_data'
train_path = '../../images/train'
test_path = '../../images/test'
valid_path = '../../images/validation'


for f in os.listdir(root_folder):

    os.mkdir(train_path + '/' + f)
    os.mkdir(test_path + '/' + f)
    os.mkdir(valid_path + '/' + f)

    img_paths = glob(root_folder + '/' + f + '/*.png')
    train_files = img_paths[:int(0.7*len(img_paths))]
    valid_files = img_paths[int(0.7*len(img_paths)):int(0.8*len(img_paths))]
    test_files = img_paths[int(0.8*len(img_paths)):]

    for file in train_files:
        copyfile(file, train_path + '/' + f + '/' + basename(file))

    for file in valid_files:
        copyfile(file, valid_path + '/' + f + '/' + basename(file))

    for file in test_files:
        copyfile(file, test_path + '/' + f + '/' + basename(file))
