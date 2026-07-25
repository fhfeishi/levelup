import os
import shutil
from random import shuffle

conservator_dir = r"F:\bianse\1280_960_dataset\conservator"
xml_dir = r"F:\bianse\1280_960_dataset\xml"

test_jpg_dir = r"F:\bianse\testdata\jpg"

name_set = {name.split('.')[0] for name in os.listdir(conservator_dir)}
train_set = {name.split('.')[0] for name in os.listdir(xml_dir)}

rest_set = name_set - train_set

rest_set_list = list(rest_set)
shuffle(rest_set_list)

test_names = rest_set_list[:100]

def get_testData(conservator_dir, test_jpg_dir, test_names):
    for name in test_names:
        jpg_filename = name + '.jpg'
        jpg_path = f"{conservator_dir}/{jpg_filename}"
        if os.path.isfile(jpg_path):
            target_path = f"{test_jpg_dir}/{jpg_filename}"
            shutil.move(jpg_path, target_path)
            # move jpg_path data to target_path
        else:
            print(jpg_path)

get_testData(conservator_dir, test_jpg_dir, test_names)
