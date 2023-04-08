import os
import math
import shutil
import random

DATASET_FOLDER = './collected-datasets/All_Oversample_224'
OUT_FOLDER = './collected-datasets'
TRAIN_FOLDER = OUT_FOLDER + '/train'
VAL_FOLDER = OUT_FOLDER + '/val'
TEST_FOLDER = OUT_FOLDER + '/test'

CATEGORIES = os.listdir(DATASET_FOLDER)

# create folders
for f in [TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER]:
    for c in CATEGORIES:
        path = '/'.join([f, c])
        os.makedirs(path, exist_ok=True)

# split train-val-test to be 80-10-10
for c in CATEGORIES:
    path = '/'.join([DATASET_FOLDER, c])
    image_files = os.listdir(path)
    random.shuffle(image_files) #shuffle the list of images

    image_count = len(image_files)
    persplit_count = math.ceil(image_count/10)
    train_split = (image_files[0:persplit_count*8], TRAIN_FOLDER)
    val_split = (image_files[persplit_count*8: persplit_count*9], VAL_FOLDER)
    test_split = (image_files[persplit_count*9: image_count], TEST_FOLDER)

    for split in [train_split, val_split, test_split]:
        files, split_folder = split

        for f in files:
            source = '/'.join([DATASET_FOLDER, c, f])
            destination = '/'.join([split_folder, c])

            shutil.move(source, destination)
