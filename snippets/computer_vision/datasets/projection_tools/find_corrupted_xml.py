import os
import xml.etree.ElementTree as ET

def find_corrupted_xml(xml_dir):
    corrupted_files = []
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            try:
                # 尝试解析 XML 文件
                path = os.path.join(xml_dir, xml_file)
                tree = ET.parse(path)
            except ET.ParseError as e:
                # 捕获解析错误
                print(f"解析错误在文件：{xml_file}，错误详情：{e}")
                corrupted_files.append(xml_file)
            except UnicodeDecodeError as e:
                # 捕获编码错误
                print(f"编码错误在文件：{xml_file}，错误详情：{e}")
                corrupted_files.append(xml_file)
            except Exception as e:
                # 捕获其他类型的错误
                print(f"其他错误在文件：{xml_file}，错误详情：{e}")
                corrupted_files.append(xml_file)

    return corrupted_files

# 使用示例
xml_dir = r'F:\bianse\1280_960_dataset\xml'
corrupted_files = find_corrupted_xml(xml_dir)
if corrupted_files:
    print("乱码或损坏的 XML 文件列表:")
    for file in corrupted_files:
        print(file)
# else:
#     print("没有发现乱码或损坏的 XML 文件。")
