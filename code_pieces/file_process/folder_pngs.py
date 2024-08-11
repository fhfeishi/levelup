import os
import shutil
from tqdm import tqdm

def folder_pngs(img_root, dir_name=1, png_num=8):
    
    files = [f for f in os.listdir(img_root) if f.endswith('.png')]

    groups = {}
    for file in files:
        prefix = file.rsplit('_', 1)[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(file)

    for prefix, file_list in groups.items():
        if len(file_list) == png_num:  
            num_folder = os.path.join(img_root, str(dir_name))
            os.makedirs(num_folder, exist_ok=True)
            for file in file_list:
                shutil.move(os.path.join(img_root, file), os.path.join(num_folder, file))
            dir_name += 1
        # else:
        #     print(prefix)

# png_num 头部12 侧面8 底部16

img_root = r"E:\xiamen_未整理\0528\0528-2.5D-头部凹坑0.15mm"
folder_pngs(img_root, dir_name=1, png_num=12)
