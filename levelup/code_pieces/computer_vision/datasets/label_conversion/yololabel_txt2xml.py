import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import PIL.Image as Image
from tqdm import tqdm 

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def yolo_to_voc(txt_file, img_file, xml_file, class_names):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    img = Image.open(img_file)
    width, height = img.size

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = os.path.basename(os.path.dirname(img_file))
    ET.SubElement(annotation, 'filename').text = os.path.basename(img_file)
    ET.SubElement(annotation, 'path').text = img_file

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'

    ET.SubElement(annotation, 'segmented').text = '0'

    for line in lines:
        elements = line.strip().split()
        class_id = int(elements[0])
        x_center, y_center, w, h = map(float, elements[1:])

        xmin = int((x_center - w / 2) * width)
        ymin = int((y_center - h / 2) * height)
        xmax = int((x_center + w / 2) * width)
        ymax = int((y_center + h / 2) * height)

        object = ET.SubElement(annotation, 'object')
        ET.SubElement(object, 'name').text = class_names[class_id]
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'difficult').text = '0'

        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    xml_str = prettify(annotation)
    with open(xml_file, 'w', encoding='utf-8') as f:
        f.write(xml_str)

# 示例使用
txt_folder = r'D:\Ddesktop\ppt\work\luoshuan0516\dataset-txts'  # YOLO标注文件夹路径
img_folder = r'D:\Ddesktop\ppt\work\luoshuan0516\dataset-jpgs'  # 图像文件夹路径
xml_folder = r'D:\Ddesktop\ppt\work\luoshuan0516\dataset-xmls'  # 生成的VOC XML文件夹路径
class_names = ['class_a', 'class_b', 'class_c', 'class_d', 'class_e', 'class_f']  # 类别名称


if not os.path.exists(xml_folder):
    os.makedirs(xml_folder)

for txt_file in tqdm(os.listdir(txt_folder)):
    if txt_file.endswith('.txt'):
        img_file = os.path.join(img_folder, txt_file.replace('.txt', '.jpg'))
        xml_file = os.path.join(xml_folder, txt_file.replace('.txt', '.xml'))

        if os.path.exists(img_file):
            yolo_to_voc(os.path.join(txt_folder, txt_file), img_file, xml_file, class_names)
        else:
            print(f"Image file {img_file} does not exist for {txt_file}")
