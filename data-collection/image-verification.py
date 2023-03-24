import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import shutil


def imshow(image, category):
    image = np.array(image)
    
    plt.imshow(image)
    plt.title(category)
    plt.axis('off')
    plt.show()

def verify():
    REMOVE_DEST_FOLDER = 'removed-images'

    with open('image-verification-input', 'r') as f:
        # get inputs for the program
        # dataset root: the root directory of the dataset that the verification program is working on
        # folders: the remaining folders in the root directory that the verification has not finished
        # next_img: the next image to verify
        lines = f.readlines()
        if len(lines) == 2:
            dataset_root, folders= lines
            next_img = None
        else:
            dataset_root, folders, next_img = lines

        dataset_root = dataset_root[:-1] # get rid of the newline
        folders = folders[:-1].split(' ') # to list


    processed_folders = [] # a folder of images has been processed will be added to this list
    start_id = 0
    if next_img:
        first_folder_dir = dataset_root + '/' + folders[0]
        images = os.listdir(first_folder_dir)
        for i,img in enumerate(images):
            if next_img == img:
                start_id=i
                break


    # iterate over the remaining folder
    for folder in folders:
        folder_dir = dataset_root + '/' + folder

        # iterate over images in the folder
        images = os.listdir(folder_dir)
        for i, img_name in enumerate(images):
            if (i<start_id):
                continue

            print(f'Image {i} out of the folder of {len(images)} images')
            img_dir = folder_dir + '/' + img_name

            # open image and print image category
            image = Image.open(img_dir)
            imshow(image, folder)

            # confirm removal of the image
            yes = input("Remove? (y/n): ")
            if yes[0].lower()=='y':
                destination = REMOVE_DEST_FOLDER + "/" + folder
                os.makedirs(destination, exist_ok=True)

                shutil.move(img_dir, destination)
                print("Removed ", img_name)
            else:
                print("Kept ", img_name)

            # confirm the continuation
            yes = input("Next Image? (y/n): ")
            if yes[0].lower()=='n':
                if (i+1==len(images)):
                    processed_folders.append(folder)
                    next_to_process = ''
                else:
                    next_to_process = images[i+1]

                left = [folder for folder in folders if folder not in processed_folders]
                with open("image-verification-input", "w") as f:
                    f.write(dataset_root+"\n")
                    f.write(' '.join(left)+"\n")
                    f.write(next_to_process)

                return
                

        start_id = 0
        processed_folders.append(folder)


verify()

    


