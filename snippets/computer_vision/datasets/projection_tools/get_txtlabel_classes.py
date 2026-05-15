import os

def get_total_classes(txt_folder):
    class_ids = set()

    # 遍历txt_folder目录中的所有文件
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            with open(os.path.join(txt_folder, txt_file), 'r') as file:
                for line in file:
                    # 假设类别ID是每行的第一个值
                    class_id = int(line.split()[0])
                    class_ids.add(class_id)

    total_classes = len(class_ids)
    return total_classes, class_ids

# 示例使用
txt_folder = r"D:\ddesktop\luoshuan0516\trainval\labels\trainval"  # 修改为txt_folder的实际路径
total_classes, class_ids = get_total_classes(txt_folder)
print(f"Total number of classes: {total_classes}")
print(f"Class IDs: {sorted(class_ids)}")
