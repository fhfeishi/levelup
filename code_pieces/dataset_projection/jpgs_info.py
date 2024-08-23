from PIL import Image
import numpy as np
import os
import tqdm
from collections import defaultdict


def jpgs_info(jpgs_dir, writeTxt=False):
    size_count = defaultdict(int)  # 用于统计各个尺寸的图片数量
    for im in tqdm.tqdm(os.listdir(jpgs_dir)):
        if im.endswith('.jpg'):
            img_path = f"{jpgs_dir}/{im}"
            image = Image.open(img_path)
            size = image.size
            size_count[size] += 1
        else:
            print("not jpg img:", im)
    if writeTxt:
        with open(f"{os.path.dirname(jpgs_dir)}/jpgsinfo.txt", 'w') as file:
            file.write("图片尺寸和数量： \n")
            for size, count in size_count.items():
                file.write(f"尺寸 {size[0]}x{size[1]}: {count} 张\n")
    else:
        print("图片尺寸和数量： \n")
        for size, count in size_count.items():
            print(f"size {size[0]}x{size[1]}: {count}")

jpgs_dir = r"F:\bianse\normaldata\conservator_normal"
jpgs_info(jpgs_dir, True)
