import os
import xml.etree.ElementTree as ET
from tqdm import tqdm 
from glob import glob
import shutil
# txts 0,1  -> head  head_with_helmet
# xmls head, helmet; ...  -> head  head_with_helmet

# filename  --


# # ##step1 
# # move to norm_filesystem
# def norm_file_system(data_root, save_root):
#     os.makedirs(f'{save_root}/images', exist_ok=True)
#     os.makedirs(f'{save_root}/labels', exist_ok=True)
#     img_path = glob(f'{data_root}/**/*',recursive=True) 
#     for impath in tqdm(img_path):
#         if impath.endswith(('.jpg', '.png')):
#             save_path = f'{save_root}/images/{os.path.basename(impath)}'
#             shutil.copyfile(impath, save_path)
#             print(f'move file {os.path.basename(impath)} done!')
#         # elif impath.endswith(('.txt', '.xml')):
#         #     save_path = f'{save_root}/labels/{os.path.basename(impath)}'
#         #     shutil.copyfile(impath, save_path)
#         #     print(f'move file {os.path.basename(impath)} done!')
#         else:
#             continue
            
# data_root = r"D:\ddesktop\helmet\dataset_to_merge\archive4-good\helm\helm"
# save_root = r"D:\ddesktop\helmet\helmetDataset\1_"
# norm_file_system(data_root=data_root, save_root=save_root)

# ##step2  check dataset  image-labels

# def clean_files(data_roots):
#     for data_root in data_roots:
#         img_folder = os.path.join(data_root, 'images')
#         label_folder = os.path.join(data_root, 'labels')

#         # img_paths = glob(os.path.join(img_folder, '*.[jp][np]g')) + glob(os.path.join(img_folder, '*.png'))
#         # label_paths = glob(os.path.join(label_folder, '*.[tx][mt]'))
#         img_paths = glob(os.path.join(img_folder, '*.jpg')) + glob(os.path.join(img_folder, '*.png'))
#         label_paths =  glob(os.path.join(label_folder, '*.txt')) + glob(os.path.join(label_folder, '*.xml'))
#         # 创建一个集合存储所有标注文件的基名（不带后缀）
#         label_basenames = {os.path.splitext(os.path.basename(label_path))[0] for label_path in label_paths}
#         image_basenames = {os.path.splitext(os.path.basename(img_path))[0] for img_path in img_paths}
#         print(f'{len(label_basenames) = }') # 0
#         print(f'{len(image_basenames) = }')  # ok
        
#         # 检查图片文件
#         for img_path in tqdm(img_paths):
#             name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
#             if name_without_ext not in label_basenames:
#                 # 如果没有对应的标注文件，则删除图片
#                 # os.remove(img_path)
#                 print(f'Deleted image without label: {os.path.basename(img_path)}')
#                 continue

#         # 检查标注文件
#         for label_path in tqdm(label_paths):
#             name_without_ext = os.path.splitext(os.path.basename(label_path))[0]
#             if name_without_ext not in {os.path.splitext(os.path.basename(img))[0] for img in img_paths}:
#                 # 如果没有对应的图片，则删除标注文件
#                 # os.remove(label_path)
#                 print(f'Deleted label without image: {os.path.basename(label_path)}')
#                 continue
#     return label_basenames - image_basenames
# # 指定子数据集的根目录
# data_roots = [
#     # r"D:\ddesktop\helmet\helmetDataset\1_",
#     r"D:\ddesktop\helmet\helmetDataset\2_",
#     r"D:\ddesktop\helmet\helmetDataset\3_",
#     # r"D:\ddesktop\helmet\helmetDataset\4_"
# ]
# print(clean_files(data_roots=data_roots))


# # ##step3 norm-filename
# import uuid
# def unify_filenames(data_roots):

    
#     for data_root in data_roots:
#         img_folder = os.path.join(data_root, 'images')
#         label_folder = os.path.join(data_root, 'labels')

#         # img_paths = glob(os.path.join(img_folder, '*.[jp][np]g')) + glob(os.path.join(img_folder, '*.png'))
#         # label_paths = glob(os.path.join(label_folder, '*.[tx][mt]'))
#         img_paths = glob(os.path.join(img_folder, '*.jpg')) + glob(os.path.join(img_folder, '*.png'))
#         label_paths =  glob(os.path.join(label_folder, '*.txt')) + glob(os.path.join(label_folder, '*.xml'))

#         for img_path in tqdm(img_paths):
#             # 生成唯一的随机字符串作为文件名
#             unique_id = str(uuid.uuid4())
#             new_img_name = f"{unique_id}.jpg"  # 或根据需要选择扩展名
#             img_dir = os.path.dirname(img_path)

#             # 重命名图片
#             new_img_path = os.path.join(img_dir, new_img_name)
#             os.rename(img_path, new_img_path)
#             # shutil.copyfile()
#             print(f'Renamed image: {new_img_name}')

#             # 处理对应的标注文件
#             corresponding_label = None
#             name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
#             for label_path in label_paths:
#                 if os.path.basename(label_path).startswith(name_without_ext):
#                     corresponding_label = label_path
#                     break

#             # 保持标注文件后缀名不变
#             if corresponding_label:
#                 label_extension = os.path.splitext(corresponding_label)[1]  # 获取原标注文件的后缀
#                 new_label_name = f"{unique_id}{label_extension}"  # 保持原后缀
#                 label_dir = os.path.dirname(corresponding_label)
#                 new_label_path = os.path.join(label_dir, new_label_name)
#                 os.rename(corresponding_label, new_label_path)
#                 # shutil.copyfile()
#                 print(f'Renamed label: {new_label_name}')


# # 指定子数据集的根目录
# data_roots = [
#     # r"D:\ddesktop\helmet\helmetDataset\1_",
#     # r"D:\ddesktop\helmet\helmetDataset\2_",
#     r"D:\ddesktop\helmet\helmetDataset\3_",
#     # r"D:\ddesktop\helmet\helmetDataset\4_"
# ]
# unify_filenames(data_roots)


# #  sub_dataset/images jpg png, labels xml txt
# # txt to xml
from xml.dom.minidom import parseString
from PIL import Image

# def create_voc_xml(filename, width, height, objects, output_folder, class_names=['b56c3', '8076e']):
#     annotation = ET.Element('annotation')
#     ET.SubElement(annotation, 'folder').text = output_folder
#     ET.SubElement(annotation, 'filename').text = filename
#     size = ET.SubElement(annotation, 'size')
#     ET.SubElement(size, 'width').text = str(width)
#     ET.SubElement(size, 'height').text = str(height)
#     ET.SubElement(size, 'depth').text = '3'  # 假设图像深度为3 (RGB)
#     for obj in objects:
#         obj_elem = ET.SubElement(annotation, 'object')
#         ET.SubElement(obj_elem, 'name').text = class_names[obj['class_id']]
#         ET.SubElement(obj_elem, 'pose').text = 'Unspecified'
#         ET.SubElement(obj_elem, 'truncated').text = '0'
#         ET.SubElement(obj_elem, 'difficult').text = '0'
#         bndbox = ET.SubElement(obj_elem, 'bndbox')
#         ET.SubElement(bndbox, 'xmin').text = str(obj['xmin'])
#         ET.SubElement(bndbox, 'ymin').text = str(obj['ymin'])
#         ET.SubElement(bndbox, 'xmax').text = str(obj['xmax'])
#         ET.SubElement(bndbox, 'ymax').text = str(obj['ymax'])
    
#     xml_str = ET.tostring(annotation)
#     xml_pretty = parseString(xml_str).toprettyxml()
#     xml_filename = os.path.join(output_folder, filename.replace('.jpg', '.xml'))
#     with open(xml_filename, 'w') as f:
#         f.write(xml_pretty)

# def convert_txt_to_voc(txt_folder, img_folder, output_folder, class_names=['b56c3', '8076e']):
#     os.makedirs(output_folder, exist_ok=True)
#     for txt_file in tqdm(os.listdir(txt_folder)):
#         if txt_file.endswith('.txt'):
#             txt_path = os.path.join(txt_folder, txt_file)
#             img_path = os.path.join(img_folder, txt_file.replace('.txt', '.jpg'))
            
#             if not os.path.exists(img_path):
#                 print(f"Image file {img_path} does not exist.")
#                 continue
            
#             # 读取图像以获取其尺寸
#             img = Image.open(img_path)
#             image_width, image_height = img.size
#             objects = []
            
#             with open(txt_path, 'r') as file:
#                 for line in file:
#                     parts = line.strip().split()
#                     if len(parts) < 5:
#                         continue  # Skip incomplete lines
#                     class_id = int(parts[0])  # Class ID remains an integer
#                     # Convert normalized coordinates back to pixel coordinates
#                     x_center = float(parts[1])
#                     y_center = float(parts[2])
#                     width = float(parts[3])
#                     height = float(parts[4])
#                     xmin = int((x_center - width / 2) * image_width)
#                     ymin = int((y_center - height / 2) * image_height)
#                     xmax = int((x_center + width / 2) * image_width)
#                     ymax = int((y_center + height / 2) * image_height)
#                     objects.append({'class_id': class_id, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
            
#             create_voc_xml(txt_file.replace('.txt', '.jpg'), image_width, image_height, objects, output_folder, class_names)

# txt_folder = r'D:\ddesktop\helmet\helmetDataset\4_\labels'
# img_folder = r'D:\ddesktop\helmet\helmetDataset\4_\images'
# output_folder = r'D:\ddesktop\helmet\helmetDataset\4_\xmls'
# os.makedirs(output_folder, exist_ok=True) # 1 2 4
# convert_txt_to_voc(txt_folder=txt_folder, img_folder=img_folder, output_folder=output_folder)


# def update_class_labels(xml_folder, class_mapping):
#     for xml_file in tqdm(os.listdir(xml_folder)):
#         if xml_file.endswith('.xml'):
#             xml_path = os.path.join(xml_folder, xml_file)
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#             # 查找所有的 <name> 标签并替换
#             for obj in root.findall('object'):
#                 name_elem = obj.find('name')
#                 if name_elem is not None:
#                     current_class = name_elem.text
#                     if current_class in class_mapping:
#                         name_elem.text = class_mapping[current_class]
#                         print(f'Updated {current_class} to {class_mapping[current_class]} in {xml_file}')
#             # 保存修改后的 XML 文件
#             tree.write(xml_path)

# 定义类名映射 #1#
# class_mapping = {
#     '8076e': '9076e',  # helmet
#     'b56c3': 'b56c9'   # head
# }
# #2#
# class_mapping = {
#     '8076e': '9076e',  # helmet
#     'b56c3': 'b56c9'   # head
# }
# # # #3#
# class_mapping = {
#     'head_with_helmet': '9076e',  # helmet
#     'head': 'b56c9'   # head
# }

# # #4#
# class_mapping = {
#     '8076e': '9076e',  # helmet
#     'b56c3': 'b56c9'   # head
# }

# xml_folder = r'D:\ddesktop\helmet\helmetDataset\3_\xmls'
# update_class_labels(xml_folder, class_mapping)

# # 仅保留head  helmet标注
# def filter_annotations(xml_folder, save_folder, allowed_classes):
#     os.makedirs(save_folder, exist_ok=True)
#     for xml_file in os.listdir(xml_folder):
#         if xml_file.endswith('.xml'):
#             xml_path = os.path.join(xml_folder, xml_file)
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#             objects = root.findall('object')
#             filtered_objects = []

#             # 过滤不需要的标注框
#             for obj in objects:
#                 name_elem = obj.find('name')
#                 if name_elem is not None and name_elem.text in allowed_classes:
#                     filtered_objects.append(obj)

#             # 清空当前的 <object> 元素并添加保留的元素
#             for obj in root.findall('object'):
#                 root.remove(obj)
#             for obj in filtered_objects:
#                 root.append(obj)

#             # 保存修改后的 XML 文件到新目录
#             new_xml_path = os.path.join(save_folder, xml_file)
#             tree.write(new_xml_path)
#             print(f'Saved filtered XML: {new_xml_path}')

# # 设置参数
# xml_folder = r'D:\ddesktop\helmet\helmetDataset\xmls'  # 原始 XML 文件夹
# save_folder = r'D:\ddesktop\helmet\helmetDataset\xmls'  # 另存目录
# # allowed_classes = ['head', 'head_with_helmet']  # 保留的标注框类型
# allowed_classes = ['9076e', 'b56c9']  # 保留的标注框类型


# filter_annotations(xml_folder, save_folder, allowed_classes)


# # # 删除空的标注数据
# def delete_empty_annotations(xml_folder):
#     for xml_file in tqdm(os.listdir(xml_folder)):
#         if xml_file.endswith('.xml'):
#             xml_path = os.path.join(xml_folder, xml_file)
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#             objects = root.findall('object')
#             # 如果没有标注框，删除该文件
#             if not objects:
#                 os.remove(xml_path)
#                 print(f'Deleted empty XML: {xml_path}')
# # 设置参数
# xml_folder = r'D:\ddesktop\helmet\helmetDataset\xmls'  # 原始 XML 文件夹

# delete_empty_annotations(xml_folder)




# # # clean  dataset
# # # 删除没有标注的图片
# def clean_files(data_roots):
#     for data_root in data_roots:
#         img_folder = os.path.join(data_root, 'images')
#         label_folder = os.path.join(data_root, 'xmls')

#         # img_paths = glob(os.path.join(img_folder, '*.[jp][np]g')) + glob(os.path.join(img_folder, '*.png'))
#         # label_paths = glob(os.path.join(label_folder, '*.[tx][mt]'))
#         img_paths = glob(os.path.join(img_folder, '*.jpg')) + glob(os.path.join(img_folder, '*.png'))
#         label_paths =  os.listdir(label_folder)
#         # 创建一个集合存储所有标注文件的基名（不带后缀）
#         label_basenames = {os.path.splitext(os.path.basename(label_path))[0] for label_path in label_paths}
#         image_basenames = {os.path.splitext(os.path.basename(img_path))[0] for img_path in img_paths}
#         print(f'{len(label_basenames) = }') # 0
#         print(f'{len(image_basenames) = }')  # ok
        
#         # 检查图片文件
#         for img_path in tqdm(img_paths):
#             name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
#             if name_without_ext not in label_basenames:
#                 # 如果没有对应的标注文件，则删除图片
#                 # os.remove(img_path)
#                 print(f'Deleted image without label: {os.path.basename(img_path)}')
#                 continue

#         # # 检查标注文件
#         # for label_path in tqdm(label_paths):
#         #     name_without_ext = os.path.splitext(os.path.basename(label_path))[0]
#         #     if name_without_ext not in {os.path.splitext(os.path.basename(img))[0] for img in img_paths}:
#         #         # 如果没有对应的图片，则删除标注文件
#         #         # os.remove(label_path)
#         #         print(f'Deleted label without image: {os.path.basename(label_path)}')
#         #         continue
#     return label_basenames - image_basenames
# # 指定子数据集的根目录
# data_roots = [
#     # r"D:\ddesktop\helmet\helmetDataset\1_",
#     r"D:\ddesktop\helmet\helmetDataset\2_",
#     r"D:\ddesktop\helmet\helmetDataset\3_",
#     # r"D:\ddesktop\helmet\helmetDataset\4_"
# ]
# print(clean_files(data_roots=data_roots))



# 根据xml移动jpg  --> dataset
def get_dataset(data_roots, save_root):
    save_imgdir = f'{save_root}/images'
    save_xmldir = f'{save_root}/xmls'
    for data_root in data_roots:
        imgdir = f'{data_root}/images'
        xmldir = f'{data_root}/xmls'
        for xmls in tqdm(os.listdir(xmldir)):
            if xmls.endswith('.xml'):
                xmls_path = f'{xmldir}/{xmls}'
                xmls_savepath = f'{save_xmldir}/{xmls}'
                print(f'{xmls = }')
                shutil.copyfile(xmls_path, xmls_savepath)
                imgs = xmls.replace('.xml', '.jpg')
                imgs_path = f'{imgdir}/{imgs}'
                imgs_savepath = f'{save_imgdir}/{imgs}'
                print(f'{imgs = }')
                shutil.copyfile(imgs_path, imgs_savepath)
                
data_roots = [
    r"D:\ddesktop\helmet\helmetDataset\1_",
    # r"D:\ddesktop\helmet\helmetDataset\2_",
    r"D:\ddesktop\helmet\helmetDataset\3_",
    r"D:\ddesktop\helmet\helmetDataset\4_"
]
save_root = r"D:\ddesktop\helmet\helmetDataset"
get_dataset(data_roots=data_roots, save_root=save_root)

# # final check  
imgs_dir = r"D:\ddesktop\helmet\helmetDataset\images"
xmls_dir = r"D:\ddesktop\helmet\helmetDataset\xmls"

imgs_name = {name.split('.')[0] for name in os.listdir(imgs_dir) if name.endswith('.jpg')}
xmls_name = {name.split('.')[0] for name in os.listdir(xmls_dir) if name.endswith('.xml')}
print(f'{len(imgs_name) = }')
print(f'{len(xmls_name) = }')
print(imgs_name - xmls_name)
