import os
import shutil

data_folder = r'F:\bianse\normaldata\conservator_normal'
label_folder = r'F:\bianse\dataset\xml'

data_nameset = {name.split(".")[0] for name in os.listdir(data_folder)}
label_nameset = {name.split(".")[0] for name in os.listdir(label_folder)}

rest_nameset = data_nameset - label_nameset

print("data len:", len(data_nameset))  # 2106
print("label len:", len(label_nameset))  # 320
print("restdata len:", len(rest_nameset))  #1786 


def copyimg(data_folder, label_nameset, new_dir):
    for name in label_nameset:
        name = name + '.jpg'
        data_path = f'{data_folder}/{name}'
        if os.path.isfile(data_path):
            new_path = f"{new_dir}/{name}"
            shutil.copy2(data_path, new_path)
            print(f"文件 {name} 已复制到 {new_dir}")
            
new_dir = r"F:\bianse\dataset\jpg"
copyimg(data_folder, label_nameset, new_dir)    

