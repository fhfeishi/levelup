import os

jpg_folder = r'D:\Ddesktop\ppt\work\luoshuan0516\dataset-jpgs'  # 替换为你的 jpg 图片文件夹路径
txt_folder = r'D:\Ddesktop\ppt\work\luoshuan0516\dataset-txts'  # 替换为你的 xml 标注文件夹路径

jpg_name_set = {name.split(".")[0] for name in os.listdir(jpg_folder)}
txt_name_set = {name.split(".")[0] for name in os.listdir(txt_folder)}

print(jpg_name_set-txt_name_set)