# img_folder下  xx_normal.png  xx_shape.png 移动到  img_folder/xx/ 目录下
import os
import shutil
from tqdm import tqdm

img_root = r"F:\cy0606\0605-2.5D-头部划痕-无套膜"

def organize_images_by_common_field(img_folder):
    # 遍历 img_folder 中的所有图片文件名
    for name in tqdm(os.listdir(img_folder)):
        if name.endswith('.png'):
            if '--------_' in name:
                common_part = name.split('--------_')[1].rsplit('_', 1)[0]
            else:
                # 获取文件名的公共部分（去掉最后一个下划线后的部分）
                common_part = name.rsplit('_', 1)[0]
            # 创建新的子文件夹路径
            new_folder_path = os.path.join(img_folder, common_part)
            # 如果子文件夹不存在，则创建
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            # 构造源文件路径和目标文件路径
            src_path = os.path.join(img_folder, name)
            dst_path = os.path.join(new_folder_path, name)
            # 移动文件到新的子文件夹
            shutil.move(src_path, dst_path)

# 调用函数
organize_images_by_common_field(img_root)



