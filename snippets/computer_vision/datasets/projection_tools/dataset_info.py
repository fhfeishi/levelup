import os
import json
from PIL import Image

# ----------------json标注的图像分割数据集----------------------
json_folder = r"F:\jyz_dataset\datadata\jyz_json"
# 统计变量
jyz_count = 0
zb_count = 0
# 统计字典
json_file_count = 0
# 遍历 JSON 文件夹
for file_name in os.listdir(json_folder):
    if file_name.endswith('.json'):
        json_file_count += 1
        file_path = os.path.join(json_folder, file_name)
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 提取每个标签及其数量
        shapes = data.get('shapes', [])
        for shape in shapes:
            label = shape.get('label')
            if label == 'jyz':
                jyz_count += 1
            elif label == 'zb':
                zb_count += 1
print("标注为 'jyz' 的目标个数:", jyz_count)
print("标注为 'zb' 的目标个数:", zb_count)
print("json count:", json_file_count)

# ----------jpg-size: jpg-count:------------------#
# JPEG文件夹路径
jpeg_folder = r'F:\jyz_dataset\datadata\jpg2250pad'
# 统计字典
resolution_count = {}
# 遍历JPEG文件夹
for file_name in os.listdir(jpeg_folder):
    if file_name.endswith('.jpg'):
        file_path = os.path.join(jpeg_folder, file_name)
        # 打开图像文件
        image = Image.open(file_path)
        # 获取图像分辨率
        resolution = image.size
        # 统计对应的分辨率和图像数目
        if resolution in resolution_count:
            resolution_count[resolution] += 1
        else:
            resolution_count[resolution] = 1
# 输出统计结果
for resolution, count in resolution_count.items():
    print("分辨率:", resolution, " 图片数目:", count)
