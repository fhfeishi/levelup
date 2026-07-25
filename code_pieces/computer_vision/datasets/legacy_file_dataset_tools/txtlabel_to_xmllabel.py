import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from PIL import Image

def create_voc_xml(filename, width, height, objects, class_names, output_folder):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = output_folder
    ET.SubElement(annotation, 'filename').text = filename

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'  # 假设图像深度为3 (RGB)

    for obj in objects:
        obj_elem = ET.SubElement(annotation, 'object')
        ET.SubElement(obj_elem, 'name').text = class_names[obj['class_id']]
        ET.SubElement(obj_elem, 'pose').text = 'Unspecified'
        ET.SubElement(obj_elem, 'truncated').text = '0'
        ET.SubElement(obj_elem, 'difficult').text = '0'
        bndbox = ET.SubElement(obj_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(obj['xmin'])
        ET.SubElement(bndbox, 'ymin').text = str(obj['ymin'])
        ET.SubElement(bndbox, 'xmax').text = str(obj['xmax'])
        ET.SubElement(bndbox, 'ymax').text = str(obj['ymax'])

    xml_str = ET.tostring(annotation)
    xml_pretty = parseString(xml_str).toprettyxml()

    xml_filename = os.path.join(output_folder, filename.replace('.jpg', '.xml'))
    with open(xml_filename, 'w') as f:
        f.write(xml_pretty)

def convert_txt_to_voc(txt_folder, img_folder, output_folder, class_names):
    os.makedirs(output_folder, exist_ok=True)

    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(txt_folder, txt_file)
            img_path = os.path.join(img_folder, txt_file.replace('.txt', '.jpg'))

            if not os.path.exists(img_path):
                print(f"Image file {img_path} does not exist.")
                continue

            # 读取图像以获取其尺寸
            img = Image.open(img_path)
            width, height = img.size

            objects = []
            with open(txt_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    xmin, ymin, xmax, ymax = map(int, parts[1:5])
                    objects.append({'class_id': class_id, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

            create_voc_xml(txt_file.replace('.txt', '.jpg'), width, height, objects, class_names, output_folder)

# 类别名称
class_names = ['a', 'b', 'c', 'd', 'e', 'f']

# 文件夹路径
txt_folder = 'path_to_txt_folder'
img_folder = 'path_to_image_folder'
output_folder = 'path_to_output_folder'

convert_txt_to_voc(txt_folder, img_folder, output_folder, class_names)

