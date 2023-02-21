import os
import pandas as pd

CATEGORIES = {
    'metals_and_plastic':1, 
    'other': 2, 
    'non_recyclable': 3,
    'glass': 4,
    'paper': 5,
    'bio': 6,
    'unknown': 7
}

trashnet_path = './collected-datasets/TrashNet'
subfolders = os.listdir(trashnet_path)

data = []
for subfolder in subfolders:
    cur_path = '/'.join([trashnet_path, subfolder])
    category_id = CATEGORIES[subfolder]
    for img_name in os.listdir(cur_path):
        img_path = '/'.join([cur_path, img_name])
        data.append((img_path, category_id))

df = pd.DataFrame(data)
csv_file = '/'.join([trashnet_path, 'labels.csv'])
df.to_csv(csv_file, header=None, index=False)    
    
    


