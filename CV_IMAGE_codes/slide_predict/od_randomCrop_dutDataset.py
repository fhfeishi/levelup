# cut dataset

import numpy as np
from PIL import Image
import cv2
import random
import xml.etree.ElementTree as ET
import os

def pil_to_cv2(pil_img):
    cv2_img = np.array(pil_img)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    return cv2_img

def cv2_to_pil(cv2_img):
    pil_image = Image.fromarray(cv2_img)
    return pil_image

def random_crop(image, bboxes, min_crop_size=0.6):
    h, w, _ = image.shape
    while True:
        crop_w = random.uniform(min_crop_size * w, w)
        crop_h = random.uniform(min_crop_size * h, h)
        
        x1 = random.uniform(0, w - crop_w)
        y1 = random.uniform(0, h - crop_h)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        crop_bbox = [int(x1), int(y1), int(x2), int(y2)]
        
        new_bboxes = []
        for bbox in bboxes:
            bbox_x1 = max(crop_bbox[0], bbox[0])
            bbox_y1 = max(crop_bbox[1], bbox[1])
            bbox_x2 = min(crop_bbox[2], bbox[2])
            bbox_y2 = min(crop_bbox[3], bbox[3])
            
            intersection_area = max(0, bbox_x2 - bbox_x1) * max(0, bbox_y2 - bbox_y1)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            if intersection_area / bbox_area > 0.6:
                new_bboxes.append([bbox_x1 - int(x1), bbox_y1 - int(y1), bbox_x2 - int(x1), bbox_y2 - int(y1)])
        
        if len(new_bboxes) > 0:
            break
    
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    return cropped_image, new_bboxes

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes

def update_xml(xml_file, new_bboxes, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall('object'):
        root.remove(obj)
    
    for bbox in new_bboxes:
        obj = ET.SubElement(root, 'object')
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(bbox[0])
        ET.SubElement(bndbox, 'ymin').text = str(bbox[1])
        ET.SubElement(bndbox, 'xmax').text = str(bbox[2])
        ET.SubElement(bndbox, 'ymax').text = str(bbox[3])
    
    tree.write(output_file)

def process_dataset(image_dir, xml_dir, output_image_dir, output_xml_dir):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_xml_dir):
        os.makedirs(output_xml_dir)

    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(image_dir, image_file)
            xml_path = os.path.join(xml_dir, os.path.splitext(image_file)[0] + '.xml')
            
            pil_image = Image.open(image_path)
            image = pil_to_cv2(pil_image)
            bboxes = parse_xml(xml_path)
            
            cropped_image, new_bboxes = random_crop(image, bboxes)
            
            output_image_path = os.path.join(output_image_dir, image_file)
            output_xml_path = os.path.join(output_xml_dir, os.path.splitext(image_file)[0] + '.xml')
            
            # output_image_path含有中文名的话，就没有输出，也不会报错
            cv2.imwrite(output_image_path, cropped_image)
            # 可以尝试转换为PIL Image，再save

            update_xml(xml_path, new_bboxes, output_xml_path)

# 示例使用
process_dataset('path_to_images', 'path_to_xmls', 'output_images', 'output_xmls')









