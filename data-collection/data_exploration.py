from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from collections import Counter

import os

def imshow(image_np):
    """
    This function receive a numpy image and display it
    """
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

# image = Image.open('./collected-datasets/All/paper/2_199.jpg')
# np_image = np.array(image)
# imshow(np_image)

def imshow_actual_size(image_np):
    width = image_np.shape[0]
    height = image_np.shape[1]

    dpi = 80
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize)
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

def imshow_pad(PIL_image, pad=50):
    """
    This function receives a PIL image, apply 0 padding, and show
    """
    pad_transform = transforms.Pad(pad, fill=0, padding_mode="constant")
    width, height = PIL_image.size
    padded_img = pad_transform(PIL_image)
    padded_img.show()
    
# image = Image.open('./collected-datasets/All/metals_and_plastic/2_248.jpg')
# imshow_pad(image)

def im_count_distribution(train_folder, test_folder, val_folder):
    """
    This function shows the distributions of images by categories for train, test, val
    """
    labels = os.listdir(train_folder)

    train_labels = []
    val_labels = []
    test_labels = []
    for label in labels:
        # train
        path = '/'.join([train_folder, label])
        images = os.listdir(path)
        train_labels.extend([label]*len(images))

        # val
        path = '/'.join([val_folder, label])
        images = os.listdir(path)
        val_labels.extend([label]*len(images))

        # test
        path = '/'.join([test_folder, label])
        images = os.listdir(path)
        test_labels.extend([label]*len(images))

    train_dist = pd.DataFrame(Counter(train_labels).items())
    test_dist = pd.DataFrame(Counter(test_labels).items())
    val_dist = pd.DataFrame(Counter(val_labels).items())

    print("==== Train Distribution dataframe: ====")
    print(train_dist, end='\n\n')
    plt.figure(figsize=(12,5))
    xs = train_dist[0].values
    ys = train_dist[1].values
    plt.bar(xs, ys, color="#5D9C59")
    plt.title("Distribution of Images in the training dataset")
    plt.ylabel("Number of images")
    for i, x in enumerate(xs):
        plt.text(i, ys[i]//2, ys[i],ha='center')
    plt.show()

    print("==== Validation Distribution dataframe: ====")
    print(val_dist)
    print("==== Test Distribution dataframe: ====")
    print(test_dist, end='\n\n')


DATASET_FOLDER = '../data-collection/collected-datasets'
TRAIN_FOLDER = DATASET_FOLDER + '/train'
VAL_FOLDER = DATASET_FOLDER + '/val'
TEST_FOLDER = DATASET_FOLDER + '/test'
# im_count_distribution(TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER)
    
def export_im_size_distribution(train_folder, test_folder, val_folder):
    """
    This function obtains images size for all images in train, test, and val
    and export them to csv_files
    """
    labels = os.listdir(train_folder)

    images = []
    for label in labels:
        # train
        path = '/'.join([train_folder, label])
        train_images = ['/'.join([path, f]) for f in os.listdir(path)]
        images.extend(train_images)

        # val
        path = '/'.join([val_folder, label])
        val_images = ['/'.join([path, f]) for f in os.listdir(path)]
        images.extend(val_images)

        # test
        path = '/'.join([test_folder, label])
        test_images = ['/'.join([path, f]) for f in os.listdir(path)]
        images.extend(test_images)

    widths = []
    heights = []
    for image in images:
        image_np = np.array(Image.open(image))
        widths.append(image_np.shape[1])
        heights.append(image_np.shape[0])

    pd.DataFrame(widths).to_csv("./images_width.csv", index=False)
    pd.DataFrame(heights).to_csv("./images_height.csv", index=False)

# export_im_size_distribution(TRAIN_FOLDER, TEST_FOLDER, VAL_FOLDER)

def im_size_statistics(csv_data_file):
    """
    This function shows statistics of image size
    """
    # get data
    data = pd.read_csv(csv_data_file).iloc[:, 0]

    print(data.describe())

    # plot
    plt.hist(data, bins=50, color="#5D9C59")
    plt.xlabel("Image Size (pixel)")
    plt.ylabel("Number of images")
    plt.title("Distribution of Image Size")
    plt.show()

# im_size_statistics('./images_height.csv')