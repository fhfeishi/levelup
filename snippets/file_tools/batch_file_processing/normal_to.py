# normal图得到 shape1  dif   # 参数配置没有调整好。。

import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_normal_map(normal_map_path):
    """ 读取Normal图并返回numpy数组 """
    normal_map = cv2.imread(normal_map_path, cv2.IMREAD_UNCHANGED)
    normal_map = normal_map.astype(np.float32) / 255.0
    normal_map = normal_map * 2 - 1  # 将范围从 [0, 1] 转换到 [-1, 1]
    return normal_map

def compute_diffuse_rf(normal_map):
    """ 计算DiffuseRF，通常是法线向量与光源方向的点积 """
    light_dir = np.array([0, 0, 1])
    diffuse_rf = np.dot(normal_map, light_dir)
    diffuse_rf = np.clip(diffuse_rf, 0, 1)
    return diffuse_rf

def compute_gloss_ratio(normal_map):
    """ 计算GlossRatio，通常与表面法线的变化有关 """
    gloss_ratio = np.linalg.norm(np.gradient(normal_map), axis=0)
    gloss_ratio = np.clip(gloss_ratio, 0, 1)
    return gloss_ratio

def compute_shape1(normal_map):
    """ 计算Shape1，假设为法线的X分量 """
    shape1 = (normal_map[:, :, 0] + 1) / 2  # 将范围从 [-1, 1] 转换到 [0, 1]
    return shape1

def compute_shape2(normal_map):
    """ 计算Shape2，假设为法线的Y分量 """
    shape2 = (normal_map[:, :, 1] + 1) / 2  # 将范围从 [-1, 1] 转换到 [0, 1]
    return shape2

def compute_specular(normal_map):
    """ 计算Specular，通常与反射光强度有关 """
    view_dir = np.array([0, 0, 1])
    reflect_dir = 2 * np.dot(normal_map, view_dir[:, np.newaxis]) * normal_map - view_dir
    specular = np.dot(reflect_dir, view_dir)
    specular = np.clip(specular, 0, 1)
    return specular

def plot_image(image, title, cmap='viridis'):
    """ 绘制图像 """
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.show()

def main(normal_map_path):
    normal_map = read_normal_map(normal_map_path)

    diffuse_rf = compute_diffuse_rf(normal_map)
    plot_image(diffuse_rf, 'DiffuseRF')

    gloss_ratio = compute_gloss_ratio(normal_map)
    plot_image(gloss_ratio, 'GlossRatio')

    shape1 = compute_shape1(normal_map)
    plot_image(shape1, 'Shape1')

    shape2 = compute_shape2(normal_map)
    plot_image(shape2, 'Shape2')

    specular = compute_specular(normal_map)
    plot_image(specular, 'Specular')

if __name__ == "__main__":
    normal_map_path = "D:\Ddesktop\ppt\work\missing-2.5D-30NG-0518\IMG_9E710037_2024-04-10_14-20-45\IMG_9E710037_2024-04-10_14-20-45_Normal.png"  # 替换为你的Normal图路径
    main(normal_map_path)
