import os
from PIL import Image
import numpy as np

# 定义源目录和目标目录
source_directory = r'E:\xiamen_已分类\cri_划痕_头_3d\顶部划痕-3D-71\顶部划痕-3D-亮度图'
target_directory = r'E:\xiamen_已分类\cri_划痕_头_3d\顶部划痕-3D-71\顶部划痕-3D-亮度图-crop2500'

# 确保目标目录存在，如果不存在则创建
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# 目标尺寸
target_width = 2867
target_height = 1948
half_target_width = target_width // 2
# 固定中心线位置
fixed_center = 2381

# 遍历源目录中的所有png文件
for filename in os.listdir(source_directory):
    if filename.endswith('.png') or filename.endswith('.bmp'):
        file_path = os.path.join(source_directory, filename)

        # 打开图片
        with Image.open(file_path) as img:
            # 确保图片尺寸符合预期
            if img.size[0] != 4096 or img.size[1] != 1948:
                print(f"Image {filename} has unexpected dimensions and was skipped.")
                continue

            # 计算裁剪的开始和结束列
            start_col = max(0, fixed_center - half_target_width)
            end_col = start_col + target_width

            # 确保不超出图像右边界
            if end_col > img.width:
                end_col = img.width
                start_col = max(0, img.width - target_width)

            # 裁剪图片
            cropped_img = img.crop((start_col, 0, end_col, target_height))

            # 构建目标文件路径
            target_path = os.path.join(target_directory, filename)

            # 保存裁剪后的图片
            cropped_img.save(target_path)
            print(f"Image {filename} was successfully centered at {fixed_center}, cropped, and saved to {target_path}.")
    else:
        print(f"No non-black pixels found in {filename}.")
