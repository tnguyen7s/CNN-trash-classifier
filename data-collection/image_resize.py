from torchvision import transforms
from PIL import Image
import numpy as np
import os
def resize_to_224(PIL_img):
    """
    This function receives a PIL image and resize it into (224,224)

    Returns:
    =======
    PIL Image
    """
    # padding if width or height is smaller than 224
    width = PIL_img.size[0]
    height = PIL_img.size[1]
    left_pad = 0
    right_pad = 0
    top_pad = 0
    bottom_pad = 0
    if (width<224):
        left_pad = int((224-width)/2)
        right_pad = 224 - width-left_pad
    
    if (height<224):
        top_pad = int((224-height)/2)
        bottom_pad = 224-height-top_pad

    pad_transform = transforms.Pad((left_pad, top_pad, right_pad, bottom_pad))
    transformed_image = pad_transform(PIL_img)

    # scaling big image to small image
    transformed_image = transformed_image.resize((224,224))
    return transformed_image

    # imshow_actual_size(np.array(PIL_img))
    # imshow_actual_size(np.array(transformed_image))

# resize_to_224(Image.open('./collected-datasets/All/metals_and_plastic/2_248.jpg'))


def resize_images(folder):
    """
    This function resizes images defined in `folder` and saves them in another folder
    """
    parent_out_folder = folder + " 224"
    labels = os.listdir(folder)
    for label in labels:
        out_folder = parent_out_folder+"/"+label
        os.makedirs(out_folder, exist_ok=True)

        in_folder = folder + "/"+label
        original = os.listdir(in_folder)
        for im_name in original:
            path = '/'.join([in_folder, im_name])
            im = Image.open(path)

            transformed_im = resize_to_224(im)
            path = '/'.join([out_folder, im_name])
            transformed_im.save(path)
 

resize_images('./collected-datasets/All_Oversample/')
