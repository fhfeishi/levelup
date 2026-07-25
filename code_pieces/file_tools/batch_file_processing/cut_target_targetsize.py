import cv2
import numpy as np
from PIL import Image

def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    return open_cv_image

def detect_circle(image):
    # 使用阈值化来创建二值图像
    _, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    # Image.fromarray(binary_image).show()
    # 使用Canny边缘检测
    edges = cv2.Canny(binary_image, 50, 150)

    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 拟合轮廓为圆形
    (x, y), radius = cv2.minEnclosingCircle(max_contour)
    center = (int(x), int(y))
    radius = int(radius)

    return center, radius


def crop_to_square(image, center, size):
    x, y = center
    half_size = size // 2

    # 计算裁剪区域
    left = max(0, x - half_size)
    right = min(image.shape[1], x + half_size)
    top = max(0, y - half_size)
    bottom = min(image.shape[0], y + half_size)

    if right - left < size:
        x = image.shape[1] // 2
        left = max(0, x - half_size)
        right = min(image.shape[1], x + half_size)
    if bottom - top < size:
        y = image.shape[0] // 2
        top = max(0, y - half_size)
        bottom = min(image.shape[0], y + half_size)

    # 裁剪图像
    cropped_image = image[top:bottom, left:right]  # h,w

    # cropped_image_array = np.array(cropped_image)

    # 如果裁剪区域不符合目标尺寸，进行填充
    # if cropped_image.shape[0] < size or cropped_image.shape[1] < size:
    #     new_image = np.zeros((size, size), dtype=np.uint8)
    #     new_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
    #     return new_image
    # else:
    #     return cropped_image
    return cropped_image, left, top, right, bottom

def process_image(image_path, source_root, output_root, target_size):
    # 读取图像
    image_pil = Image.open(image_path).convert('L')  # 确保图像是灰度图
    image = pil_to_cv2(image_pil)

    relative_path = os.path.relpath(image_path, source_root)
    output_path = os.path.join(output_root, relative_path)
    # 确保目标目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # output_path = f"{output_root}/{os.path.basename(image_path)}"
    
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return

    center, radius = detect_circle(image)
    if center is not None:
        cropped_image, left, top, right, bottom = crop_to_square(image, center, target_size)

        # # 保存裁剪后的图像
        # cv2.imwrite(output_path, cropped_image)
        target_crop = Image.fromarray(cropped_image)
        target_crop.save(output_path)

        with open(log_file, 'a') as log:
            log.write(f"{os.path.basename(image_path)}, {left}, {top}, {right}, {bottom}\n")

        print(f"Cropped image saved to {output_path}")
    else:
        print(f"No circle detected in the image: {image_path}")

# 示例使用
# image_path = 'path_to_your_image.png'
source_root = 'E:/xiamen_已分类/cri_划痕_头_3d'
output_root = 'E:/xiamen_已分类/cri_划痕_头_3d_crop2500'
# output_root = 'G:/cy0516/dibuhuahen_25d_'
import os
os.makedirs(output_root, exist_ok=True)
target_size = 2800  # 目标正方形尺寸
from glob import glob
from tqdm import tqdm

# image_paths = glob("G:/cy0516/cemianposun_25d_0516/*_Normal.png")
# image_paths = glob("G:/cy0516/cri_划痕_底_25d/*/*/*_Normal.png")+\
#               glob("G:/cy0516/cri_划痕_底_25d/*/*_Normal.png")

image_paths = glob("E:/xiamen_已分类/cri_划痕_头_3d/*/*/*.bmp")

log_file = 'E:/xiamen_已分类/cri_划痕_头_3d_crop2500/log.txt'
for image_path in tqdm(image_paths):
    process_image(image_path, source_root, output_root, target_size)