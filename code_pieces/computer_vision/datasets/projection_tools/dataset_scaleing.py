# object detection
import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def resize_img(img_dir, save_dir, target_size=(1280, 960)):
    for name in tqdm(os.listdir(img_dir)):
        if name.endswith('.jpg'):
            img_path = f"{img_dir}/{name}"
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize(target_size, Image.ANTIALIAS)
            img_resized.save(f"{save_dir}/{name}")

# #1# jpg-xml voc
def resize_dataset_odvoc(img_dir, xml_dir, save_dir, target_size=(1280, 960)):
    # 创建保存图片和 XML 文件的目录
    save_img_dir = os.path.join(save_dir, 'jpg')
    save_xml_dir = os.path.join(save_dir, 'xml')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_xml_dir, exist_ok=True)

    # 遍历 XML 文件
    for xml_file in tqdm(os.listdir(xml_dir)):
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图片文件名
        imgfilename = xml_file.split('.')[0] + '.jpg'
        img_path = os.path.join(img_dir, imgfilename)
        if not os.path.isfile(img_path):
            print(imgfilename)
            continue  # 如果图片不存在，则跳过

        # 加载图片
        img = Image.open(img_path)
        original_size = img.size

        # 调整图片大小
        img_resized = img.resize(target_size, Image.ANTIALIAS)

        # 更新 XML 文件中的分辨率信息
        root.find('size/width').text = str(target_size[0])
        root.find('size/height').text = str(target_size[1])

        # 更新 bounding box 坐标
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # 计算新坐标
            xmin = int(xmin * target_size[0] / original_size[0])
            ymin = int(ymin * target_size[1] / original_size[1])
            xmax = int(xmax * target_size[0] / original_size[0])
            ymax = int(ymax * target_size[1] / original_size[1])

            # 更新坐标值
            bndbox.find('xmin').text = str(xmin)
            bndbox.find('ymin').text = str(ymin)
            bndbox.find('xmax').text = str(xmax)
            bndbox.find('ymax').text = str(ymax)

        # 保存调整大小后的图片
        img_resized.save(os.path.join(save_img_dir, imgfilename))

        # 保存更新后的 XML 文件
        tree.write(os.path.join(save_xml_dir, xml_file))

def recover_dataset_odvoc(img_dir, xml_dir, baseImg_dir, save_dir):
    # 遍历 XML 文件
    for xml_file in tqdm(os.listdir(xml_dir)):
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图片文件名   --original_size
        imgfilename = xml_file.split('.')[0] + '.jpg'
        img_path = os.path.join(img_dir, imgfilename)
        if not os.path.isfile(img_path):
            print(imgfilename)
            continue  # 如果图片不存在，则跳过
        # 加载图片
        img = Image.open(img_path)
        original_size = img.size

        # --target_size
        target_size = Image.open(f"{baseImg_dir}/{imgfilename}").size
        
        # 调整图片大小
        img_resized = img.resize(target_size, Image.ANTIALIAS)

        # # 更新 XML 文件中的分辨率信息
        # root.find('size/width').text = str(target_size[0])
        # root.find('size/height').text = str(target_size[1])

        # 更新 bounding box 坐标
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # 计算新坐标
            xmin = int(xmin * target_size[0] / original_size[0])
            ymin = int(ymin * target_size[1] / original_size[1])
            xmax = int(xmax * target_size[0] / original_size[0])
            ymax = int(ymax * target_size[1] / original_size[1])

            # 更新坐标值
            bndbox.find('xmin').text = str(xmin)
            bndbox.find('ymin').text = str(ymin)
            bndbox.find('xmax').text = str(xmax)
            bndbox.find('ymax').text = str(ymax)

        # # 保存调整大小后的图片
        # img_resized.save(os.path.join(img_dir, imgfilename))

        # 保存更新后的 XML 文件
        tree.write(os.path.join(save_dir, xml_file))

if __name__  == '__main__':
    resize_dataset_odvoc(img_dir=r"F:\bianse\dataset\jpg",
                         xml_dir=r"F:\bianse\dataset\xml",
                         save_dir=r"F:\bianse\1280_960_dataset",
                         target_size=(1280,960))
    # resize_img(img_dir=r"F:\bianse\normaldata\conservator_normal", save_dir=r"F:\bianse\1280_960_dataset\conservator")
    
    # recover_dataset_odvoc(img_dir=r"F:\bianse\1280_960_dataset\conservator",
    #                       xml_dir=r"F:\bianse\1280_960_dataset\xml",
    #                       baseImg_dir=r"F:\bianse\normaldata\conservator_normal",
    #                       save_dir=r"F:\bianse\dataset\xml")
    
    
    