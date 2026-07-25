import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

def crop_images_from_annotations(xml_dir, jpg_dir, cls_set_dir):
    # 遍历XML目录中的所有文件
    for xml_file in tqdm(os.listdir(xml_dir)):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取对应的JPG文件
            jpg_file_name = xml_file[:-3] + "jpg"
            jpg_path = os.path.join(jpg_dir, jpg_file_name)
            
            if not os.path.exists(jpg_path):
                print(f"Image file {jpg_path} not found.")
                continue
            
            # 打开图像文件
            image = Image.open(jpg_path)
            
            # 遍历所有的目标对象
            for obj in root.findall('object'):
                label = obj.find('name').text  # 中文标签
                # 获取边界框坐标
                xmlbox = obj.find('bndbox')
                x1 = int(xmlbox.find('xmin').text)
                y1 = int(xmlbox.find('ymin').text)
                x2 = int(xmlbox.find('xmax').text)
                y2 = int(xmlbox.find('ymax').text)
                
                # 裁剪图像
                cropped_image = image.crop((x1, y1, x2, y2))
                
                # 处理中文路径问题
                save_dir = os.path.join(cls_set_dir, label)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, jpg_file_name)
                
                # 保存裁剪后的图像
                cropped_image.save(save_path)
                print(f"Saved cropped image to {save_path}")
        else:
            print('gg')

# 调用函数
xml_dir = r'F:\bianse\1280_960_dataset\xml'
jpg_dir = r'F:\bianse\1280_960_dataset\jpg'
cls_set_dir = r'F:\bianse\code\mobilenetv3\clsdata'
if not os.path.exists(cls_set_dir):
    os.makedirs(cls_set_dir)
crop_images_from_annotations(xml_dir, jpg_dir, cls_set_dir)
