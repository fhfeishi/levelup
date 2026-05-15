import json
import numpy as np
from PIL import Image

def get_palette(png_path, unique_palette=True, stamp=True, write2json=False):
    # 读取mask标签
    target = Image.open(png_path)
    # 提取调色板
    palette = target.getpalette()  # 获取调色板数据
    palette_array = np.array(palette).reshape(-1, 3)  # 调整为Nx3的数组（N个RGB颜色）  # 有可能有重复的啊
    # # 完整的包含重复的 palette_array
    if not unique_palette:
        print(palette_array)
    else:
        # 去重且保持顺序
        seen = set()
        unique_palette = []
        for color in palette_array:
            # 将颜色元组作为集合的键，以确保唯一性
            color_tuple = tuple(color)
            if color_tuple not in seen:
                seen.add(color_tuple)
                unique_palette.append(color)

        unique_palette_array = np.array(unique_palette)
        if stamp is True:
            print("Unique palette array without duplicates:")
            print(unique_palette_array)

        if write2json is True:
            json_str = json.dumps(unique_palette_array)
            with open("palette.json", "w") as f:
                f.write(json_str)

# target = np.array(target)
# print(target)

if __name__ == '__main__':
    png_path = "../data/image(4).png"