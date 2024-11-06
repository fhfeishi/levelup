import os
import xml.etree.ElementTree as ET
from collections import defaultdict



def analysis_xml(xml_dir, output_file=None):
    # 字典存储各个类别的计数
    class_count = defaultdict(int)
    # 字典存储各个分辨率的图像数量
    resolution_count = defaultdict(int)
    # 字典存储各种尺寸的标注框数量
    bbox_size_count = defaultdict(int)

    # 遍历指定目录下的所有 XML 文件
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            # 构建完整的文件路径
            path = os.path.join(xml_dir, xml_file)
            # 解析 XML 文件
            tree = ET.parse(path)
            root = tree.getroot()
            
            # 提取图像的分辨率
            width = root.find('size/width').text
            height = root.find('size/height').text
            resolution = f'{width}x{height}'
            resolution_count[resolution] += 1

            # 遍历所有的 object 标签
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_count[class_name] += 1
                
                # 获取 bounding box 尺寸
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                bbox_size = f'{bbox_width}x{bbox_height}'
                bbox_size_count[bbox_size] += 1

    # 打印结果
    print("类别及其数量:")
    for cls, count in class_count.items():
        print(f'{cls}: {count}')
    
    print("\n分辨率及其图像数量:")
    for res, count in resolution_count.items():
        # print(f'{res}: {count}')
        continue
    
    print("\n标注框尺寸及其数量:")
    for size, count in bbox_size_count.items():
        # print(f'{size}: {count}')
        continue

    if (output_file is not None) and (os.path.isfile(output_file)):
        # 打开文件并写入数据
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("类别及其数量:\n")
            for cls, count in class_count.items():
                file.write(f'{cls}: {count}\n')
            
            file.write("\n分辨率及其图像数量:\n")
            for res, count in resolution_count.items():
                file.write(f'{res}: {count}\n')

            file.write("\n标注框尺寸及其数量:\n")
            for size, count in bbox_size_count.items():
                file.write(f'{size}: {count}\n')



xml_dir = r"F:\switch\switch_data\1021newdata\dataset_train\xmls"

txt_path = os.path.join(os.path.dirname(xml_dir), "datasetinfo.txt")
# 调用函数
analysis_xml(xml_dir, txt_path)

