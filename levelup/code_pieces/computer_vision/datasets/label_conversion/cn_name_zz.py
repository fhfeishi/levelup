import os
import re
from tqdm import tqdm

import unicodedata

def normalize_to_ascii(name):
    """将中文字符替换为字母或数字，确保文件名为全英文字符"""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'\W+', '', name)  # 移除非字母和非数字字符
    return name

def rename_files_with_chinese(folder_path):
    for root, _, files in os.walk(folder_path):
        for file_name in tqdm(files):
            # 获取文件名和后缀
            name, ext = os.path.splitext(file_name)
            # 替换文件名中的中文字符为字母或数字
            new_name = normalize_to_ascii(name)
            if new_name != name:
                old_file_path = os.path.join(root, file_name)
                new_file_name = new_name + ext
                new_file_path = os.path.join(root, new_file_name)
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} -> {new_file_path}')
# 示例使用
jpg_folder = r'D:\Ddesktop\ppt\work\luoshuan0516\dataset-jpgs'  # 替换为你的 jpg 图片文件夹路径
txt_folder = r'D:\Ddesktop\ppt\work\luoshuan0516\dataset-txts'  # 替换为你的 xml 标注文件夹路径

rename_files_with_chinese(jpg_folder)
rename_files_with_chinese(txt_folder)
