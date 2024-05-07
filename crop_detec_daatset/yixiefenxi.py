import os
import cv2
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

# crop_size = 640*640,   crop 截取高分辨率原图 的  带有目标区域的 块，还得同步更新得到的块对应的标注xml文件

#  crop     [[1]   [2]   [3]   [4]  [5]  ...]
#  #0#  crop 的移动策略，只是移动的话，目标的大小是不固定的，不太好设置步长，感觉还是很呆。
#           
#        设置一个 threshold， 满足这个threshold，       
#  --ok--   iou应该可以算一下  box-h box-w  640 --->get a inside point(x, y)  
#                           --> crop (x-320, y+320)还得在图片内部啊 

#  #1#  crop 仅包含 100%的 box    --更新xml  
#       最少的框，框住所有的box，

#  #2#  crop包含<100% box  --更新xml  
#       最少的框，框住所有的box



jpg_folder = r"D:\codespace\creations\projection\yolov8-pytorch\VOCdevkit\VOC2007\JPEGImages"
xml_folder = r"D:\codespace\creations\projection\yolov8-pytorch\VOCdevkit\VOC2007\Annotations"


jpg_folder_out = r"D:\codespace\creations\projection\yolov8-pytorch\VOCdevkit\VOC2007\jpg_crop"
os.makedirs(jpg_folder_out, exist_ok=True)
xml_folder_out = r"D:\codespace\creations\projection\yolov8-pytorch\VOCdevkit\VOC2007\xml_crop"
os.makedirs(xml_folder_out, exist_ok=True) 

def crop_image(image, bboxes, crop_size=640):
    height, width, channels = image.shape[:-2]
    # 这个死循环有点呆
    while True:
        xmin = np.random.randint(0, width - crop_size)
        ymin = np.random.randint(0, height - crop_size)
        crop_image = image[ymin:ymin + crop_size, xmin:xmin + crop_size]
        crop_bboxes = [(bbox[0] - xmin, bbox[1] - ymin, bbox[2] - xmin, bbox[3] - ymin) for bbox in bboxes]
        crop_area = sum([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in crop_bboxes])
        total_area = crop_size * crop_size
        if crop_area / total_area >= 0.6:
            return crop_image, crop_bboxes

def crop_dec_dataset(image_folder, xml_folder, image_folder_out, xml_folder_out, crop_size=640):
    for filename in tqdm(os.listdir(image_folder)):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            xml_path = os.path.join(xml_folder, filename.replace('.jpg', '.xml').replace('.jpeg', '.xml').replace('.png', '.xml'))

            # Load image and XML
            image = cv2.imread(image_path)
            if image is None:
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = root.findall('object')
            bboxes = [(int(obj.find('bndbox').find('xmin').text), int(obj.find('bndbox').find('ymin').text),
                      int(obj.find('bndbox').find('xmax').text), int(obj.find('bndbox').find('ymax').text)) for obj in objects]

            # Crop image
            cropped_image, crop_bboxes = crop_image(image, bboxes, crop_size)
            image_basename = os.path.basename(filename)
            save_path = os.path.join(image_folder_out, image_basename)
            cv2.imwrite(save_path, cropped_image)

             # Crop XML
            if os.path.exists(xml_path):
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    # Crop bounding box
                    new_xmin, new_ymin, new_xmax, new_ymax = crop_bbox(bbox, crop_bboxes)

                    # Update bounding box coordinates
                    bbox.find('xmin').text = str(new_xmin)
                    bbox.find('ymin').text = str(new_ymin)
                    bbox.find('xmax').text = str(new_xmax)
                    bbox.find('ymax').text = str(new_ymax)

                # Save cropped XML
                xml_basename = os.path.basename(xml_path)
                save_xml_path = os.path.join(xml_folder_out, xml_basename)
                tree.write(save_xml_path)
        




if __name__ == '__main__':








