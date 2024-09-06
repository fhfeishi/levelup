import os
from tqdm import tqdm
from PIL import Image
import shutil
# 1
# # 去掉空格\中文 
# # filename.replace(' ', '')
# def del_kongge(dir_path):
#     # 获取目录中的所有文件和文件夹名
#     names = os.listdir(dir_path)
#     # 使用 tqdm 来显示进度条
#     for name in tqdm(names): 
#         # 新文件名去除空格
#         new_name = name.replace(' ', '')
#         # 如果新旧文件名不同，则重命名
#         if new_name != name:
#             # 构建完整的旧文件路径和新文件路径
#             old_path = os.path.join(dir_path, name)
#             new_path = os.path.join(dir_path, new_name)
#             # 重命名操作
#             os.rename(old_path, new_path)
#             print(f'rename {old_path} to {new_path}')

# dir_path = r"F:\bianse\switch\labels\train"
# del_kongge(dir_path)

# # 2
# # JPG2jpg
# def get_jpg(jpg_dir, save_dir):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)
        
#     for name in tqdm(os.listdir(jpg_dir)):
#         if name.endswith('.JPG'):
#             jpg_path = f"{jpg_dir}/{name}"
#             image = Image.open(jpg_path)
#             new_name = name[:-3] + "jpg"
#             image.save(f"{save_dir}/{new_name}", "JPEG")
#         elif name.endswith('.jpg'):
#             jpg_path = f"{jpg_dir}/{name}"
#             image = Image.open(jpg_path)
#             image.save(f"{save_dir}/{name}")
#         else:
#             print(name)
#             continue
# jpg_dir = r"F:\bianse\switch\images\val"
# save_dir = r"F:\bianse\switch\val_dataset\images"            
# get_jpg(jpg_dir, save_dir)

# # 3
# # move txt-file
# def copytxt(txt_dir, save_dir):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)
    
#     for file in tqdm(os.listdir(txt_dir)):
#         if file.endswith('.txt'):
#             shutil.copy2(f"{txt_dir}/{file}", f"{save_dir}/{file}")
#         else:
#             print(file)
#             continue
# txt_dir = r"F:\bianse\switch\labels\train"
# save_dir = r"F:\bianse\switch\train_dataset\txts"
# copytxt(txt_dir, save_dir)

# # 3
# # file-name cn -> en
# from uuid import uuid4
# def filename_cn2en(image_dir, txt_dir):
#     # 确保新的目录存在
#     new_image_dir = os.path.join(os.path.dirname(image_dir), 'new_images')
#     new_txt_dir = os.path.join(os.path.dirname(txt_dir), 'new_txts')
#     os.makedirs(new_image_dir, exist_ok=True)
#     os.makedirs(new_txt_dir, exist_ok=True)

#     # 记录原始和新的文件名映射
#     name_map = {}

#     # 处理图片文件
#     for image_name in tqdm(os.listdir(image_dir)):
#         if image_name.endswith('.jpg'):
#             new_name = str(uuid4()) + '.jpg'
#             name_map[image_name] = new_name
#             src_path = os.path.join(image_dir, image_name)
#             dst_path = os.path.join(new_image_dir, new_name)
#             shutil.copy2(src_path, dst_path)

#     # 处理标注文件，确保与图片文件名对应
#     for txt_name in os.listdir(txt_dir):
#         if txt_name.endswith('.txt'):
#             base_name = txt_name[:-4] + '.jpg'  # 将.txt替换为.jpg查找对应的图片
#             if base_name in name_map:
#                 new_txt_name = name_map[base_name][:-4] + '.txt'
#                 src_path = os.path.join(txt_dir, txt_name)
#                 dst_path = os.path.join(new_txt_dir, new_txt_name)
#                 shutil.copy2(src_path, dst_path)

#     print("Renaming complete. New files are in 'new_images' and 'new_txts' directories.")

# image_dir = r"F:\bianse\switch\val_dataset\images"
# txt_dir = r"F:\bianse\switch\val_dataset\txts"
# filename_cn2en(image_dir, txt_dir)

# # dataset  image-label check   --ok
# image_dir = r"F:\bianse\switch\val_dataset\images"
# label_dir = r"F:\bianse\switch\val_dataset\txts"
# image_nameSet = {name.split('.')[0] for name in os.listdir(image_dir)}
# label_nameSet = {name.split('.')[0] for name in os.listdir(label_dir)}
# print(image_nameSet-label_nameSet)
# print(label_nameSet-image_nameSet)

# # 4
# # get len(classes)
# def extract_annotation_info(txt_dir):
#     category_set = set()  # 使用集合来存储独特的类别，避免重复
#     # 遍历目录中的所有文件
#     for filename in os.listdir(txt_dir):
#         if filename.endswith('.txt'):  # 确保只处理txt文件
#             file_path = os.path.join(txt_dir, filename)
#             with open(file_path, 'r') as file:
#                 for line in file:
#                     parts = line.strip().split()
#                     if len(parts) >= 5:  # 确保这是一个有效的标注行
#                         class_id = parts[0]  # 类别ID是每行的第一个元素
#                         category_set.add(class_id)  # 添加到集合中
#     return category_set
# txt_dir = r"F:\bianse\switch\train_dataset\new_txts"
# print(extract_annotation_info(txt_dir))
# # train: {'1', '0'}
# # val: {'1', '0'}


# 4
#  txt_label to 
import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from PIL import Image
import uuid
import random

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
            image_width, image_height = img.size

            objects = []
            with open(txt_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # Skip incomplete lines
                    class_id = int(parts[0])  # Class ID remains an integer
                    # Convert normalized coordinates back to pixel coordinates
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    xmin = int((x_center - width / 2) * image_width)
                    ymin = int((y_center - height / 2) * image_height)
                    xmax = int((x_center + width / 2) * image_width)
                    ymax = int((y_center + height / 2) * image_height)
                    
                    objects.append({'class_id': class_id, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

                    
                    create_voc_xml(txt_file.replace('.txt', '.jpg'), image_width, image_height, objects, class_names, output_folder)


def generate_unique_short_ids(num_ids, length=4):
    unique_ids = set()
    while len(unique_ids) < num_ids:
        # 生成一个 UUID
        random_uuid = uuid.uuid4().hex  # 获取无破折号的十六进制字符串形式
        # 从 UUID 中随机抽取四个字符
        start_index = random.randint(0, len(random_uuid) - length)
        short_id = random_uuid[start_index:start_index + length]
        # 添加到集合中以保证唯一性
        unique_ids.add(short_id)
    return list(unique_ids)
# # 类别名称
class_names = generate_unique_short_ids(2,5)
# print(class_names) # ['b56c3', '8076e']

# 文件夹路径 1
txt_folder_1 = r'F:\bianse\switch\val_dataset\new_txts'
img_folder_1 = r'F:\bianse\switch\val_dataset\new_images'
output_folder_1 = r'F:\bianse\switch\val_dataset\new_xmls'
if not os.path.exists(output_folder_1):
    os.makedirs(output_folder_1)
convert_txt_to_voc(txt_folder_1, img_folder_1, output_folder_1, class_names)

# 文件夹路径 2
txt_folder_2 = r'F:\bianse\switch\train_dataset\new_txts'
img_folder_2 = r'F:\bianse\switch\train_dataset\new_images'
output_folder_2 = r'F:\bianse\switch\train_dataset\new_xmls'
if not os.path.exists(output_folder_2):
    os.makedirs(output_folder_2)
convert_txt_to_voc(txt_folder_2, img_folder_2, output_folder_2, class_names)
