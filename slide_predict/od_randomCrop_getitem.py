import random 
import numpy as np
import cv2
from PIL import Image
import xml.etree.ElementTree as ET

def pil_to_cv2(pil_img):
    cv2_img = np.array(pil_img)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    return cv2_img

def cv2_to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)
    return pil_img

# 还不能很好的解决：超大目标
# 目标-bbox  远大于  crop_size 的话，可能并不能裁剪到该目标
def od_randomCrop(image, bboxes, crop_size=(640,640), min_objectInArea_ratio=0.6):
    h, w, _ = image.shape
    cw, ch = crop_size
    assert h >= ch and w >= cw, f"Image size must be at least {cw}x{ch}"

    while True:
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)
        x2 = x1 + cw
        y2 = y1 + ch

        crop_bbox = [x1, y1, x2, y2]

        # 记录相对于crop的bbox
        new_bboxes = []
        for bbox in bboxes:
            bbox_x1 = max(crop_bbox[0], bbox[0])
            bbox_y1 = max(crop_bbox[1], bbox[1])
            bbox_x2 = min(crop_bbox[2], bbox[2])
            bbox_y2 = min(crop_bbox[3], bbox[3])

            intersection_area = max(0, bbox_x2 - bbox_x1) * max(0, bbox_y2 - bbox_y1)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # 如果 bbox_area * min_objectInArea_ratio > crop_size
            # 目标检测数据集 xml-bbox  area-size： count：  
            # 对于超大目标+小目标的，代码还需要额外处理
            if bbox_area * min_objectInArea_ratio > cw * ch:
                # 这里就减小 ratio 裁剪 超大目标的部分区域， 
                # 网络训练的好的话，能否通过这些部分区域学习到超大目标的全部区域？
                new_ratio = (cw*ch)/bbox_area - 0.1
                if intersection_area / bbox_area > new_ratio:
                    new_bboxes.append([bbox_x1 - x1, bbox_y1 - y1, bbox_x2 - x1, bbox_y2 - y1])
            # bbox_area * min_objectInArea_ratio =< crop_size
            else:
                if intersection_area / bbox_area > min_objectInArea_ratio:
                    new_bboxes.append([bbox_x1 - x1, bbox_y1 - y1, bbox_x2 - x1, bbox_y2 - y1])

        if len(new_bboxes) > 0:
            cropped_image = image[y1:y2, x1:x2]
            return cropped_image, new_bboxes

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.fimd('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])



