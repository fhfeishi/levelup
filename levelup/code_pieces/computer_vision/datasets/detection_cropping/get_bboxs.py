# 获取bbox的一些信息  bbox的宽度、高度
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

xml_folder = r"D:\codespace\creations\projection\yolov8-pytorch\VOCdevkit\VOC2007\Annotations"

def get_bbox_info(xml_file):
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 遍历文件中的每个对象
    for obj in root.findall('object'):
        # 获取bbox信息
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # 计算宽度和高度
        width = xmax - xmin
        height = ymax - ymin
        
        # 打印信息或进行其他处理
        print(f'File: {xml_file}, Object: {obj.find("name").text}, Width: {width}, Height: {height}')


if __name__ == '__main__':

    for file in tqdm(os.listdir(xml_folder)):
        if file.endswith('.xml'):
            xml_file = os.path.join(xml_folder, file)
            get_bbox_info(xml_file)  #  128*128 以内
