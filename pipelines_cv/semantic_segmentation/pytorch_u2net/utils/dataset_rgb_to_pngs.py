# 彩色rgb mask .png文件转换为 单通道 mask.png --putpalette 看上去是彩色的
# rgb_png to dataset:png
from PIL import Image
import numpy as np
import json
import os

def get_color_label_map(png_path, write_down_json=False):
    # 加载图像
    png = Image.open(png_path)
    png_data = np.array(png)
    # 获取图像中的所有唯一颜色 # 获取除了黑色背景外的其他color
    unique_colors = np.unique(png_data.reshape(-1, png_data.shape[2]), axis=0)

    color_label_map = {(0, 0, 0): 0}   # (0, 0, 0)  0 background
    # 打印唯一颜色
    print("Unique colors in the image:")
    for color in unique_colors:
        print(color)   # numpy.ndarray
        if tuple(color) in color_label_map:
            label_value = color_label_map[tuple(color)]
        else:
            label_value = len(color_label_map)
            color_label_map[tuple(color)] = label_value

        rgb_values, label_values = [], []
        for rgb, lbv in sorted(color_label_map.items(), key=lambda x: x[1]):
            rgb_values.append(rgb)
            label_values.append(lbv)

        assert label_values == list(range(len(label_values)))
    print(color_label_map)

    if write_down_json:
        with open('palette.json', 'w') as json_file:
            json.dump(color_label_map, json_file)
    return color_label_map

# {(0, 0, 0): 0, (40, 200, 30): 1, (210, 80, 60): 2}
def get_png_from_palette(png_path, rgb_to_label_value, png_folder):
    png = Image.open(png_path)  # 这里有可能会包含 alpha通道， cv2.imread() 默认是过滤alpha通道的  RGBA  A透明度通道
    png_array = np.array(png)  # h, w, c

    # 创建一个空的单通道数组，用于存储索引 shape: h, w
    indexed_png = np.zeros(png_array.shape[:2], dtype=np.uint8)


    for i in range(png_array.shape[0]):
        for j in range(png_array.shape[1]):
            rgb = tuple(png_array[i, j, :3])  # 前3个通道 rgb， 过滤掉了alpha通道
            # print(rgb)
            indexed_png[i, j] = rgb_to_label_value.get(rgb, 0)  # 0是设置的默认值，如果没找到这个rgb

    # 创建调色板 palette
    palette = []
    for i in range(256):
        if i in rgb_to_label_value.values():
            # 找出颜色索引对应的RGB值
            color = list(next(key for key, value in rgb_to_label_value.items() if value == i))
        else:
            # 默认颜色，可以设为黑色或任意颜色
            color = [0, 0, 0]
        palette.extend(color)

    # 将调色板转换成1维数组, 没有这一步就是3通道了
    palette = palette + [0] * (768 - len(palette))  # 确保调色板长度为768

    # print(indexed_png)
    new_png = Image.fromarray(indexed_png, mode='P')
    new_png.putpalette(palette)

    save_path = os.path.join(png_folder, os.path.basename(png_path))
    new_png.save(save_path)


if __name__ == '__main__': 
    # # jpg-image test
    # png_path = '../DUTS-TR/DUTS-TR-Mask/image(4).png'
    # color_label_map = get_color_label_map(png_path)
    # png_folder = '../data'
    # os.makedirs(png_folder, exist_ok=True)
    # get_png_from_palette(png_path, color_label_map, png_folder)


    from tqdm import tqdm 

    # # get color label map：
    # png_path = r"CV_PROJECTION\SOD_u2net\raw_data\DUTS-TE\DUTS-TE-Mask\image(60).png"
    # get_color_label_map(png_path)
    # # {(0, 0, 0): 0, (40, 200, 30): 1, (210, 80, 60): 2}
    
    color_label_map = {(0, 0, 0): 0, (40, 200, 30): 1, (210, 80, 60): 2}

    # rgb pngs
    png_folder = r"CV_PROJECTION\SOD_u2net\raw_data\DUTS-TR\DUTS-TR-Mask"

    # pngs :dataset
    target_png_folder = r"CV_PROJECTION\SOD_u2net\dataset\train_data\pngs"

    for png_file in tqdm(os.listdir(png_folder)):
        if png_file.endswith(".png"):
            png_path = os.path.join(png_folder, png_file)
            get_png_from_palette(png_path, color_label_map, target_png_folder)


