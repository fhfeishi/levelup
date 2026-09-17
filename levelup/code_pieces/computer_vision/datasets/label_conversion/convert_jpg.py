import os
from PIL import Image
from tqdm import tqdm
def convert_images_to_jpg(img_folder):
    # 获取文件夹下所有的文件名
    for root, _, files in os.walk(img_folder):
        for file_name in tqdm(files):
            # 检查文件的扩展名是否为需要转换的格式
            if file_name.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.gif')):
            # if file_name.endswith(('.jpeg', '.JPG', '.png', '.bmp', '.tiff', '.gif', '.PNG')):
                file_path = os.path.join(root, file_name)
                img = Image.open(file_path).convert('RGB')  # 确保图像是RGB格式

                # 构造新的文件名
                new_file_name = os.path.splitext(file_name)[0] + '.jpg'
                new_file_path = os.path.join(root, new_file_name)

                # 保存图像为.jpg格式
                if file_name.endswith('.jpg'):
                    pass
                else:
                    img.save(new_file_path, 'JPEG', quality=95)

                # 删除原始文件（如果原始文件不是.jpg格式）
                # if not file_name.endswith('.jpg'):
                if file_name.lower() != new_file_name.lower():
                    # print(file_name)
                    os.remove(file_path)
                print(f"Converted and saved {file_path} to {new_file_path}")

# 示例使用
img_folder = r'D:\Ddesktop\ppt\work\luoshuan0516\测试数据，只用来测试\images'
convert_images_to_jpg(img_folder)
