import os
import xml.etree.ElementTree as ET


def parse_xml(xml_path):
    tree = ET.parse(xml_path)  # or:  tree = ET.parse(open(xml_path, encoding='utf-8')) # 默认只读模式'r'
    root = tree.getroot()
    
    image_info = {
        'folder': root.find('folder').text,
        'filename': root.find('filename').text,
        'path': root.find('path').text,
        'width': root.find('size/width').text,
        'height': root.find('size/height').text,
        'depth': root.find('size/depth').text
    }

    annotations = []
    for obj in root.iter('object'):
        anno = {
            'name': obj.find('name').text,
            'pose': obj.find('pose').text,
            'truncated': int(obj.find('truncated').text),
            'difficult': int(obj.find('difficult').text),
            'bbox':[
                int(obj.find('bndbox/xmin').text),
                int(obj.find('bndbox/ymin').text),
                int(obj.find('bndbox/xmax').text),
                int(obj.find('bndbox/ymax').text)
            ]
        }
        annotations.append(anno)
        
    return image_info, annotations
