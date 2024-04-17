import json
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
import os

#--------------------.xml  --> .json----------------------------------------


#--------------------.json --> .xml----------------------------------------


#--------------------.json --> .png---------------------------------
jyz_json_folder = r"F:\jyz_dataset\datadata\jyz_json"
jpg_folder = r"F:\jyz_dataset\datadata\jpg"
jyz_png_folder =r"F:\jyz_dataset\datadata\jyz_png"
for jyz_json in tqdm(os.listdir(jyz_json_folder)):
    jyz_json_path = os.path.join(jyz_json_folder, jyz_json)
    # print(jyz_json_path)
    jpg_path = os.path.join(jpg_folder, jyz_json.replace('.json', '.jpg'))
    # 加载你的JSON文件
    with open(jyz_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    jpg_image = Image.open(jpg_path)
    width, height = jpg_image.size
    binary_image = np.zeros((height, width), dtype=np.uint8)
    # 遍历JSON文件中的所有对象
    for item in data['shapes']:
        if item['label'] == 'jyz':
            # 确保所有的点都是数字元组
            points = [(float(x), float(y)) for x, y in item['points']]
            # 在这里，我们需要创建一个临时的PIL图像来绘制多边形
            temp_image = Image.new('L', (width, height), 0)
            ImageDraw.Draw(temp_image).polygon(points, outline=1, fill=1)
            binary_image = np.maximum(binary_image, np.array(temp_image))
    # 将numpy数组转换为图像
    binary_image = Image.fromarray(binary_image * 255)  # 将图像数据转换为0和255
    # 保存图像
    png_path = os.path.join(jyz_png_folder, jyz_json.replace('.json', '.png'))
    # print("png_path:", png_path)
    binary_image.save(png_path)



