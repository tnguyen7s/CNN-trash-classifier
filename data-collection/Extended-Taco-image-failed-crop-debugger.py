import pandas as pd
import json
from image_crop import *

def get_successful_cropped_annot_ids(labels_file):
    # labels_file = './collected-datasets/Extended-Taco/labels.csv'

    # get file paths
    df = pd.read_csv(labels_file, header=None)
    ls = df[0].to_list()

    # get annotation ids that have been cropped
    annot_ids = []
    for path in ls:
        id = int(path.split('/')[-1].split('.')[0].split('_')[1])
        annot_ids.append(id)

    return annot_ids

def get_failed_cropped_annotations_file(annotation_files, result_annotation_file, labels_file):
    # annotation_files = ['./annotations/Extended_Taco_annotations_test.json', './annotations/Extended_Taco_annotations_train.json']
    # result_annotation_file = './annotations/Extended_Taco_annotations_failed.json'
    success_ids = get_successful_cropped_annot_ids(labels_file)

    # get annotations that have not been cropped 
    unsuccess_annots = []
    unsuccess_imgs = []
    img_map = {}
    for filepath in annotation_files:
        with open(filepath) as f:
            json_object = json.load(f)

            # get all image annotations and store in a map (image_id -> image annotation)
            for img_annot in json_object['images']:
                img_id = img_annot['id']
                img_map[img_id] = img_annot

            for an_annot in json_object['annotations']:
                if (an_annot['id'] not in success_ids):
                    img_id = an_annot['image_id'] # find the corresponding img id

                    # save image file_name in the annotation
                    img_annot = img_map[img_id]
                    an_annot['file_name'] = img_annot['file_name']

                    # recreate bbox
                    # bbox = an_annot['bbox']
                    # value1 = bbox[0]
                    # value2 = bbox[1]
                    # bbox[0] = value2
                    # bbox[1] = value1
                    # an_annot['bbox'] = bbox

                    # add trash annotation to the list
                    unsuccess_annots.append(an_annot)

                    # add image annotation to the list
                    unsuccess_imgs.append(img_annot)

    # create annotations json for unsuccessful cropped images
    json_object = {'annotations': unsuccess_annots, 'images': unsuccess_imgs}

    with open(result_annotation_file, 'w') as f:
        json.dump(json_object, f) 

def try_to_crop():
    # get_failed_cropped_annotations_file()
    annotation_filepath = './annotations/Extended_Taco_annotations_failed.json'
    src_img_folder = './Extended-Taco-dataset'
    dest_img_folder = './collected-datasets/Extended-Taco-failed-recropped'
    labels_filepath = './collected-datasets/Extended-Taco-failed-recropped/labels.csv'
    zoom = 1
    square = True
    rotate_90_couter = False
 
    # crop(src_img_folder, 'dumped/6WYHCrMXfj9cBuTxuhik2uZnCTjNcAHlK3LpcFB7.jpeg', dest_img_folder, 'metals_and_plastic', '1_2', [2650.43, 1634.55, 60.41, 48.06], zoom, square)
    crop_and_label_Taco(4, annotation_filepath, src_img_folder, dest_img_folder, labels_filepath, zoom, square, rotate_90_couter)

# get_failed_cropped_annotations_file(['./annotations/Extended_Taco_annotations_failed.json'], './annotations/Extended_Taco_annotations_failed.json', './collected-datasets/Extended-Taco-failed-recropped/labels.csv')
try_to_crop()