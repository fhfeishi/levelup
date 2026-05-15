import os
import shutil
from tqdm import tqdm

all_img_folder = r"F:\cy0523\2.5D-头部OK-215个"
target_folder = r"F:\aaaaaa-xiamen2024-ok\okz"

# all_name_set = {name.split('.')[0] for name in os.listdir(all_img_folder) if name.endswith('.png')}
# # name_set = {name.split('.')[0] for name in os.listdir(target_folder) if name.endswith('.png')}
# name_set = {name.split('.')[0].rsplit('_', 1)[0] for name in os.listdir(target_folder) if name.endswith('.png')}


# 获取目标文件夹中normal图片的basename集合
target_basename_set = {name.rsplit('_', 1)[0] for name in os.listdir(target_folder) if name.endswith('.png')}


def moveFiles_sameStr(all_img_folder, target_basename_set, target_folder):
    # 创建目标文件夹，如果不存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 遍历所有图片文件名
    for name in tqdm(os.listdir(all_img_folder)):
        if name.endswith('.png'):
            basename = name.rsplit('_', 1)[0]
            # 检查文件名的basename是否包含在target_basename_set中
            if basename in target_basename_set:
                src_path = os.path.join(all_img_folder, name)
                dst_path = os.path.join(target_folder, name)
                shutil.copyfile(src_path, dst_path)


moveFiles_sameStr(all_img_folder, target_basename_set, target_folder)