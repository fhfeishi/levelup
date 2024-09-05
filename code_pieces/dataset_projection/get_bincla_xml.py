import xml.etree.ElementTree as ET
import os


def get_bcla_xml(xml_dir, new_dir, class_="呼吸器-硅胶"):
    # 确保新目录存在
    os.makedirs(new_dir, exist_ok=True)

    # 遍历原始 XML 目录中的所有文件
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 遍历所有 object 元素，更改其 name 子元素的文本
            for obj in root.findall('object'):
                name = obj.find('name')
                name.text = class_  # 设置新的类别名称

            # 保存修改后的 XML 到新目录
            new_xml_path = os.path.join(new_dir, xml_file)
            tree.write(new_xml_path, encoding='utf-8', xml_declaration=True)


xml_dir = r"F:\bianse\1280_960_dataset\xml"
new_dir = r"F:\bianse\1280_960_dataset\2xml"

# 调用函数
get_bcla_xml(xml_dir, new_dir)