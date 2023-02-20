# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:27:01 2023

@author: Tuyen
"""
# a cropped image saved in dest_img_folder/category_name/annotation_id.jpg
# e.g.  datasets/Taco/bio/1.jpg


import json
import os
import cv2
import pandas as pd
import warnings

CATEGORIES = {
    1: 'metals_and_plastic',
    2: 'other',
    3: 'non_recyclable',
    4: 'glass',
    5: 'paper',
    6: 'bio',
    7: 'unknown'
}


# Convert Yolo bb to Coco bb
def yolo_to_coco(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    return [x1, y1, w, h]


def crop(src_img_folder, src_img_name, dest_img_folder, category_name, annotation_id, bbox, zoom, square, yolo=False):
    """
    To crop and save cropped image

    Parameters
    ----------
    # params to access image
    src_img_folder: folder contains the source image (e.g. ./Extended_Taco_dataset)
    
    src_img_name: the image name (e.g. batch1/0001.jpg)

    => img resides in ./Extended_Taco_dataset/batch1/0001.jpg
    
    
    # params to create destination image path
    
    dest_img_folder: folder for the dest image (e.g. collected-datasets/Extended-Taco)
    
    category_name: the category of the trash instance; one of the seven categories (e.g. bio)
    
    annotation_id: the id of annotation (usage: subset_num+"_"+annotat_id; e.g.1_1)
    
    => cropped img resides in collected-datasets/Extended-Taco/bio/1_1.jpg
    
    # params to crop images
    
    bbox: [x_top_left, y_top_left, width, height] for coco and normalized [x_center, y_center, width, height] for yolo
    
    zoom: zoom out or in bounding box
    
    square: cut image into square
    
    yolo: True if bbox in Yolo format
        
    Returns
    -------
    None.

    """
    # read image to crop
    src_img_path = os.path.join(src_img_folder, src_img_name)
    img = cv2.imread(src_img_path)
    
    # if image does not exist
    if (img is None):
        print('return -1')
        return -1

    # get image width and height
    img_dims = img.shape
    image_h, image_w = img_dims[0], img_dims[1]

    # prepare to crop
    x, y, width, height = bbox
    
    # convert from yolo to coco
    if (yolo):
        x, y, width, height = yolo_to_coco(x, y, width, height, image_w, image_h)
    

    # if we want square image
    if square:
        if width > height:
            y = y - (width-height)/2
            height = width
        else:
            x = x - (-width+height)/2
            width = height
    width *= zoom
    height *= zoom

    # prepare slicing index
    x1 = int(abs(x))
    x2 = int(abs(x) + abs(width))
    y1 = int(abs(y))
    y2 = int(abs(y) + abs(height))

    if (x2>image_w):
        x2 = image_w-1
    if (y2>image_h):
        y2 = image_h-1

    # crop
    try:            
        cropped_img = img[y1:y2,x1:x2]
        
        # save
        dest_img_path = os.path.join(dest_img_folder, category_name)
        os.makedirs(dest_img_path, exist_ok=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv2.imwrite(os.path.join(dest_img_folder, category_name, str(annotation_id)+".jpg"), cropped_img)
    except Exception as e:
        print("Error when cropping ", src_img_name, bbox)
        print("image shape: ", img.shape)
        print(str(y1)+":"+str(y2)+" "+str(x1)+":"+ str(x2))
        print(cropped_img)
        return -1
    
    return 0

def crop_and_label_Taco(subset_num, annotation_filepath, src_img_folder, dest_img_folder, labels_filepath, zoom, square):
    """
    crop and label Taco images
    images goes to 

    Parameters
    ----------
    subset_num : the sub dataset number (since data collected from multiple sources, each sub dataset can be assigned with a number)
    
    annotation_filepath : the file path to the annotations file of the Taco dataset (e.g. ./annotations/Extended_Taco_annotations_test.json)
    
    src_img_folder : the parent folder of the images (e.g. ./Extended_Taco_dataset)
    
    dest_img_folder : the parent folder of the cropped images sub folders (e.g. ./collected-datasets/Extended-Taco)
        
    labels_filepath : the file path to the csv file containing labels (e.g. ./collected-datasets/Extended-Taco/labels.csv)
        
    zoom: zoom out or in bounding box
 
    square: cut image into square

    Returns
    -------
    None.

    """
    
    # open annotations file
    with open(annotation_filepath) as f:
        # load the annotations
        annotations = json.load(f)

    # reading image id and image filename
    image_filenames = {}
    for an_img in annotations["images"]:
        img_id = an_img['id']
        image_filenames[img_id] = an_img['file_name']

    # for labeling
    labels_data = {'image_path': [], 'category_id': []} 

    i=0 # index
    skipped_num = 0
    for an_annotat in annotations['annotations']:
        # reading annotation
        bbox = an_annotat['bbox']
        category_id = an_annotat['category_id']
        annotat_id = str(subset_num)+"_"+str(an_annotat['id'])
        image_id = an_annotat['image_id']
        
        image_filename = image_filenames[image_id]
        category_name = CATEGORIES[category_id]
        
            
        
        
        # crop image
        if (crop(src_img_folder, 
             image_filename, 
             dest_img_folder, 
             category_name, 
             annotat_id, 
             bbox, 
             zoom, 
             square
            ) == 0):
            # add label record
            labels_data['image_path'].append('/'.join([dest_img_folder, category_name, str(annotat_id)+".jpg"]))
            labels_data['category_id'].append(category_id)
            
        
            i+=1    
        else:
            skipped_num += 1

        # printing
        if (i%100==0):
            print("Cropped: "+str(i)+" trash instances")
            
    # save labels to csv
    labels = pd.DataFrame(labels_data)
    labels.to_csv(labels_filepath, mode='a', index=False, header=False)
        
            
    # printing
    print("Cropped and labeled: "+str(i)+" trash instances")
    print("Skipped "+str(skipped_num)+" instances")

def crop_and_label_DrinkingWaste(subset_num, src_img_folder, dest_img_folder, labels_filepath, zoom, square):
    """
    crop and label DrinkingWaste dataset

    Parameters
    ----------
    subset_num : the sub dataset number (since data collected from multiple sources, each sub dataset can be assigned with a number)
    
    src_img_folder : the parent folder of the drinkingWaste dataset (e.g. DrinkingWaste dataset). subfolder is the same as the DrinkingWaste archive downloaded from Kaggle
    ---- subfolder to use: images_of_waste/Yolo_imgs
    
    dest_img_folder : the parent folder of the cropped images sub folders (e.g. ./collected-datasets/DrinkingWaste)
    
    labels_filepath : the file path to the csv file containing labels (e.g. ./collected-datasets/DrinkingWaste/labels.csv)
    
    zoom: zoom out or in bounding box
 
    square: cut image into square
    
    Returns
    -------
    None.

    """
    src_data_path = os.path.join(src_img_folder, 'Images_of_Waste', 'YOLO_imgs')
    files = os.listdir(src_data_path)
    
    annotation_txt_files = [f for f in files if f.split('.')[1]=='txt']
    image_exts = {f.split('.')[0]: f.split('.')[1] for f in files if f.split('.')[1]!='txt'}


    # for labeling
    labels_data = {'image_path': [], 'category_id': []} 
     
    for i, an_annotation_file in enumerate(annotation_txt_files):
        name=an_annotation_file.split('.')[0]
        src_img_name =  name+ "."+image_exts[name]
        
        # get annotation id
        annotat_id = str(subset_num)+"_"+str(i)
        
        # get bbox
        with open(os.path.join(src_data_path, an_annotation_file)) as f:
            line = f.readline()
            
        nums = line.split(' ')
        for i in range(len(nums)):
            nums[i] = float(nums[i])
            
        bbox = nums[1:]
                
        
        # get category info
        category_id = 1
        if ("Glass" in name):
            category_id = 4
        
        category_name = CATEGORIES[category_id]
        
        # crop image
        crop(src_data_path, src_img_name, dest_img_folder, category_name, annotat_id, bbox, zoom, square, yolo=True)
    
    
        # add label record
        labels_data['image_path'].append('/'.join([dest_img_folder, category_name, str(annotat_id)+".jpg"]))
        labels_data['category_id'].append(category_id)
        
        # printing
        if (i%100==0):
            print("Cropped: "+str(i)+" trash instances")
            
    # save labels to csv
    labels = pd.DataFrame(labels_data)
    labels.to_csv(labels_filepath, mode='a', index=False, header=False)
    print("Cropped and labeled: "+str(i)+" trash instances")

    
if __name__== '__main__':
    # annotation_filepath = './annotations/Extended_Taco_annotations_test.json'
    # src_img_folder = './Extended-Taco-dataset'
    # dest_img_folder = './collected-datasets/Extended-Taco'
    # labels_filepath = './collected-datasets/Extended-Taco/labels.csv'
    # zoom = 1
    # square = True
    
    # crop_and_label_Taco(1, annotation_filepath, src_img_folder, dest_img_folder, labels_filepath, zoom, square)


    annotation_filepath = './annotations/Extended_Taco_annotations_train.json'
    src_img_folder = './Extended-Taco-dataset'
    dest_img_folder = './collected-datasets/Extended-Taco'
    labels_filepath = './collected-datasets/Extended-Taco/labels.csv'
    zoom = 1
    square = True
 
    crop_and_label_Taco(2, annotation_filepath, src_img_folder, dest_img_folder, labels_filepath, zoom, square)
    
    
    # subset_num = 3
    # src_img_folder = './DrinkingWaste-dataset'
    # dest_img_folder = './collected-datasets/DrinkingWaste'
    # labels_filepath = './collected-datasets/DrinkingWaste/labels.csv'
    # zoom = 1
    # square = True
    # crop_and_label_DrinkingWaste(subset_num, src_img_folder, dest_img_folder, labels_filepath, zoom, square)
