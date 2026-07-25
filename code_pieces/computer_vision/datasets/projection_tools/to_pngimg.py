import os
from PIL import Image
from tqdm import tqdm 

def img2png(img_dir, target_dir):
    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历图片目录的所有文件
    for filename in tqdm(os.listdir(img_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # 构建完整的文件路径
            file_path = os.path.join(img_dir, filename)
            # 读取图片
            img = Image.open(file_path)
            # 构建目标文件路径，将原图片的扩展名改为.png
            target_file_path = os.path.join(target_dir, os.path.splitext(filename)[0] + '.png')
            # 保存图片为png格式
            img.save(target_file_path, 'PNG')
            print(f'Saved {target_file_path}')

image_dir = r"D:\ddesktop\xianlan_measure\pslabel"
save_dir = r"D:\ddesktop\xianlan_measure\bianping_dataset\pngs"

img2png(img_dir=image_dir, target_dir=save_dir)
