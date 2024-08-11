import os

def write_filenames_to_txt(directory, output_file):
    """将指定目录下所有文件的文件名写入指定的txt文件中"""
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(directory):
            for file in files:
                f.write(file + '\n')

if __name__ == "__main__":
    directory = r"E:\xiamen_已标注\5601项目训练数据\头部套膜破损2-已解析\新项目_images"  # 修改为folderA的实际路径
    output_file = r"E:\xiamen_已标注\5601项目训练数据\头部套膜破损2-已解析\新项目_images\labels2.txt"  # 修改为aaa.txt的实际路径

    write_filenames_to_txt(directory, output_file)
    print(f"All filenames from {directory} have been written to {output_file}")
