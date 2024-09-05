import os
import shutil

data_folder = r'D:\ddesktop\xianpian\tmpsaveimage'
label_folder = r'D:\ddesktop\xianpian\datadata\layer3-png'

data_nameset = {name.split(".")[0] for name in os.listdir(data_folder)}
label_nameset = {name.split(".")[0] for name in os.listdir(label_folder)}

rest_nameset = data_nameset - label_nameset

print("data len:", len(data_nameset))  # 2106      # 943:  layer1-604   layer2-188    layer3-127   ？
print("label len:", len(label_nameset))  # 320
print("restdata len:", len(rest_nameset))  #1786 

# # check
# label_noJpg_nameset = {name for name in label_nameset if name not in data_nameset}
# print(label_noJpg_nameset)


def copyimg(data_folder, label_nameset, new_dir):
    # --->  xxx.jpg
    for name in label_nameset:
        name = name + '.jpg'
        data_path = f'{data_folder}/{name}'
        if os.path.isfile(data_path):
            new_path = f"{new_dir}/{name}"
            shutil.copy2(data_path, new_path)
            print(f"文件 {name} 已复制到 {new_dir}")
        elif os.path.isfile(data_path[:-3] + "bmp"):
            new_path = f"{new_dir}/{name}"
            shutil.copy2(data_path[:-3] + "bmp", new_path)
            print(f"文件 {name} 已复制到 {new_dir}")
# new_dir = r"D:\ddesktop\xianpian\datadata\jpg-layer3"
# copyimg(data_folder, label_nameset, new_dir)    

