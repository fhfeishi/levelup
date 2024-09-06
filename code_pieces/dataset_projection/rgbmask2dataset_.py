# rgb_mask:.png  to dataset_:.png
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm 
import ast  # Import ast to safely evaluate string literals to tuples


def get_color_label_map(png_folder, write_palette_json=True):
    
    color_label_map = {(0, 0, 0): 0}   # (0, 0, 0)  0 background
    
    for name in tqdm(os.listdir(png_folder)):
        if name.endswith('.png'):
            png_path = f"{png_folder}/{name}"
            # 加载图像
            png = Image.open(png_path).convert('RGB')
            png_data = np.array(png)
            # 获取图像中的所有唯一颜色 # 获取除了黑色背景外的其他color
            unique_colors = np.unique(png_data.reshape(-1, png_data.shape[2]), 
                                      axis=0)

            # unique_colors = {tuple(color) for color in 
            # np.unique(png_data.reshape(-1, png_data.shape[2]), axis=0) 
            # if tuple(color) != (0, 0, 0)}

            # # 打印唯一颜色
            # print("Unique colors in the image:")
            for color in unique_colors:
                # print(color)   # numpy.ndarray
                if tuple(color) in color_label_map:
                    label_value = color_label_map[tuple(color)]
                else:
                    label_value = len(color_label_map)
                    color_label_map[tuple(color)] = label_value

                rgb_values, label_values = [], []
                for rgb, lbv in sorted(color_label_map.items(), 
                                       key=lambda x: x[1]):
                    rgb_values.append(rgb)
                    label_values.append(lbv)

                assert label_values == list(range(len(label_values)))
                
    print(color_label_map)  # dic {key:(0,0,0),...   value:0,1,2 ...}

    if write_palette_json:
        with open('palette.json', 'w') as json_file:
            # json.dump(color_label_map, json_file)
            json.dump({str(k): v for k, v in color_label_map.items()}, json_file)
            # json文件的 items的key不支持tuple，可以是 str, int, float, bool or None, not tuple
    
    return color_label_map

# {(0, 0, 0): 0, (40, 200, 30): 1, (210, 80, 60): 2}
def get_png_from_palette(png_folder, rgb_to_label_value, save_dir):
    os.makedirs(save_dir, exist_ok=True)  # 不存在就新建，存在就什么都不做

    # 创建调色板 palette
    palette = []
    for i in range(256):
        if i in rgb_to_label_value.values():
            # 找出颜色索引对应的RGB值 
            # # # key is tuple  ----> dict
            # color = list(next(key for key, value in rgb_to_label_value.items() 
            #                   if value == i))
            # # # key is str  ---> json
            # color = list(next(ast.literal_eval(key) for key, value in rgb_to_label_value.items() 
            #                   if value == i))
            
            # a  # it will raise a StopIteration exception if no elements satisfy the condition.
            color = list(next(ast.literal_eval(key) if isinstance(key, str) else key 
                              for key, value in rgb_to_label_value.items() 
                              if value == i)) 
            # # b
            # for key, value in rgb_to_label_value.items():
            #     if value == i:
            #         if isinstance(key, str):
            #             color = list(ast.literal_eval(key))
            #         elif isinstance(key, tuple):
            #             color = list(key)
        else:
            # 默认颜色，可以设为黑色或任意颜色
            color = [0, 0, 0]
        palette.extend(color)

    # 将调色板转换成1维数组, 没有这一步就是3通道了
    palette = palette + [0] * (768 - len(palette))  # 确保调色板长度为768
            
    for name in tqdm(os.listdir(png_folder)):
        if name.endswith('.png'):
            png_path = f"{png_folder}/{name}"
            png = Image.open(png_path).convert('RGB')  # 这里有可能会包含 alpha通道， cv2.imread() 默认是过滤alpha通道的  RGBA  A透明度通道
            png_array = np.array(png)  # h, w, c

            # 创建一个空的单通道数组，用于存储索引 shape: h, w
            indexed_png = np.zeros(png_array.shape[:2], dtype=np.uint8)


            for i in range(png_array.shape[0]):
                for j in range(png_array.shape[1]):
                    rgb = tuple(png_array[i, j, :3])  
                    # # key: str(color)
                    indexed_png[i, j] = rgb_to_label_value.get(rgb, 0)  # 0是设置的默认值，如果没找到这个rgb
                    # # key: tuple(color) 
                    indexed_png[i, j] = rgb_to_label_value.get(str(rgb), 0)  # 0是设置的默认值，如果没找到这个rgb

            # print(indexed_png)
            new_png = Image.fromarray(indexed_png, mode='P')
            new_png.putpalette(palette)  
            
            # new_png.show()

            save_path = os.path.join(save_dir, os.path.basename(png_path))
            new_png.save(save_path)
            print(f"save {name} to {save_dir} done")


if __name__ == '__main__':
    # get <dataset> color label map
    png_folder = r'E:\datasets\triplel\before' # jpg、png folder_rawdata_
    # color_label_map = get_color_label_map(png_folder)
    color_label_map = {"(0, 0, 0)": 0, "(0, 255, 0)": 1, "(255, 255, 255)": 2} # -> set_palette not
    # color_label_map = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 255, 255): 2}   # -> set_palette ok
    # set palette   -->  get target_dataset
    save_dir = r"E:\datasets\triplel\SegmentationClass"
    get_png_from_palette(png_folder, color_label_map, save_dir)


