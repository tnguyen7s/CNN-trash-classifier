# We are going to download some of the images that have prefix `dumped` in the annotations/Taco_annotations_train and annotations/Taco_annotations_test 
# by using the URLs defined in https://github.com/pedropro/TACO/blob/master/data/all_image_urls.csv
# downloaded images will be saved at Taco-dataset/dumped

import json
import pandas as pd
import requests
import os

all_urls_file = './Extended-Taco-dataset/all_image_urls.csv'
df = pd.read_csv(all_urls_file, header=None)
urls = df[1].to_list()

# prepare urls
filename_to_url_map = {}
for url in urls:
    filename = url.split("/")[-1]
    filename_to_url_map[filename] = url

# prepare images to download
annotation_files = ['./annotations/Extended_Taco_annotations_test.json', './annotations/Extended_Taco_annotations_train.json']

imgs_to_download = []
for annotation_file in annotation_files:
    with open(annotation_file) as f:
        annotations = json.load(f)
    
    for img_metadata in annotations['images']:
        parts = img_metadata['file_name'].split('/')
        if (parts[0]=='dumped'): # we don't have images in dumped folder, thats why we download 
            imgs_to_download.append(parts[1])

# download images in imgs_to_download list
dest_img_folder = 'Extended-Taco-dataset/dumped'
os.makedirs(dest_img_folder, exist_ok=True)

downloaded_imgs = os.listdir(dest_img_folder)
i=0
for img in imgs_to_download:
    if img in filename_to_url_map and img not in downloaded_imgs:
        img_url = filename_to_url_map[img]
        img_data = requests.get(img_url).content

        img_path = '/'.join([dest_img_folder, img])
        with open(img_path, "wb") as f:
            f.write(img_data)

        i+=1

        if (i%100==0):
            print('Downloaded '+str(i)+' images.')

print('Downloaded '+str(i)+' images.')

    






