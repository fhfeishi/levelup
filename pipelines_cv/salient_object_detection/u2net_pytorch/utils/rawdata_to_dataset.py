import os
import shutil
import random
from tqdm import tqdm

raw_jpgs_folder = "../raw_data/jpgs"
# print(os.path.exists(raw_jpgs_folder))
raw_pngs_folder = "../raw_data/pngs"

train_jpgs_folder = "../dataset/train_data/jpgs"
train_pngs_folder = "../dataset/train_data/pngs"

test_jpgs_folder = "../dataset/test_data/jpgs"
test_pngs_folder = "../dataset/test_data/pngs"

jpgs_name_set = {name.split('.')[0] for name in os.listdir(raw_jpgs_folder) if name.endswith('.jpg')}
pngs_name_set = {name.replace('.png', '') for name in os.listdir(raw_pngs_folder) if name.endswith('.png')}

# print(jpgs_name_set - pngs_name_set)  # set()
assert jpgs_name_set - pngs_name_set == set(), "raw data images-labels not match"

name_list = list(jpgs_name_set)
# print(name_list)  # ['name', '']
random.shuffle(name_list)
# print("len(data):", len(name_list))
# train_name_len = xx
train_name_len = int(0.9*len(name_list)) 

train_data_names = name_list[:train_name_len]
test_data_names = name_list[train_name_len:]

# move to train-data
for name in tqdm(train_data_names):
    jpg_path = f"{raw_jpgs_folder}/{name}.jpg"
    target_jpg_path = f"{train_jpgs_folder}/{name}.jpg"
    shutil.copyfile(jpg_path, target_jpg_path)

    png_path = f"{raw_pngs_folder}/{name}.png"
    target_png_path = f"{train_pngs_folder}/{name}.png"
    shutil.copyfile(png_path, target_png_path)

# move to test-data
for name in tqdm(test_data_names):
    jpg_path = f"{raw_jpgs_folder}/{name}.jpg"
    target_jpg_path = f"{test_jpgs_folder}/{name}.jpg"
    shutil.copyfile(jpg_path, target_jpg_path)

    png_path = f"{raw_pngs_folder}/{name}.png"
    target_png_path = f"{test_pngs_folder}/{name}.png"
    shutil.copyfile(png_path, target_png_path)

