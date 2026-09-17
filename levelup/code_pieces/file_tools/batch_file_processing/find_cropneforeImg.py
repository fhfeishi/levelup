import os
from PIL import Image
import numpy as np

def load_images_from_directory(directory):
    """加载目录中的所有图片"""
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):  # 假设图片格式是这些
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path).convert('L')  # 转换为灰度图
            images.append((filename, image))
    return images

def is_subimage(big_image, small_image):
    """检查small_image是否是big_image的子图"""
    big_image_np = np.array(big_image)
    small_image_np = np.array(small_image)
    
    big_h, big_w = big_image_np.shape
    small_h, small_w = small_image_np.shape
    
    for i in range(big_h - small_h + 1):
        for j in range(big_w - small_w + 1):
            if np.array_equal(big_image_np[i:i+small_h, j:j+small_w], small_image_np):
                return True
    return False

def find_cropped_images(dir_a, dir_b):
    """查找dir_b中的图片是否来自dir_a中的图片"""
    images_a = load_images_from_directory(dir_a)
    images_b = load_images_from_directory(dir_b)
    
    results = []
    for filename_b, img_b in images_b:
        found = False
        for filename_a, img_a in images_a:
            if is_subimage(img_a, img_b):
                results.append((filename_b, filename_a))
                found = True
                break
        if not found:
            results.append((filename_b, None))
    return results

if __name__ == "__main__":
    dir_a = r"E:\xiamen_已分类\cri_划痕_头_3d\0410-顶部划痕-3D-4\顶部划痕-3D-亮度图"  # 修改为dir_a的实际路径
    dir_b = r"E:\xiamen_已标注\5601项目训练数据\3D-端面-盖帽划痕\新项目_images"  # 修改为dir_b的实际路径

    results = find_cropped_images(dir_a, dir_b)

    for file_b, file_a in results:
        if file_a:
            print(f"{file_b} is a cropped image from {file_a}")
        else:
            print(f"{file_b} does not match any image in dir_a")
