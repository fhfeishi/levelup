# status: ok
from PIL import Image
import numpy as np

def analyze_image(image_path):
    # 加载图片
    img = Image.open(image_path)
    img_array = np.array(img)
    print("Image shape:", img_array.shape)  # 打印图片的尺寸和通道数

    # 确定图片模式和通道数
    print("Image mode:", img.mode)  # 打印图片模式

    # 读取像素值
    if img.mode in ['RGB', 'RGBA']:  # 处理彩色图像
        # 获取图像中的所有唯一颜色
        # 注意: 对于RGBA, 考虑整个四通道作为颜色的一部分
        if img.mode == 'RGBA':
            unique_colors = np.unique(img_array.reshape(-1, 4), axis=0)
        else:
            unique_colors = np.unique(img_array.reshape(-1, 3), axis=0)

        print("Unique colors in the image:")
        print(unique_colors)
    elif img.mode == 'L':  # 处理灰度图像
        # 获取图像中的所有唯一灰度值
        unique_gray_values = np.unique(img_array)
        print("Unique gray values in the image:")
        print(unique_gray_values)
    elif img.mode == 'P':
        # 提取调色板
        palette = img.getpalette()  # 获取调色板数据
        palette_array = np.array(palette).reshape(-1, 3)  # 调整为Nx3的数组（N个RGB颜色）

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
        print("Unique palette array without duplicates:")
        print(unique_palette_array)
        return unique_palette_array
    else:
        print("Unsupported image mode")

    return img_array

if __name__ == '__main__':
    # 使用示例
    image_path = '../data/image(4).png'  # 替换为您的图片路径
    analyze_image(image_path)
