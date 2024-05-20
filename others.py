# data = {'name': 'python', 'price': 45.6}
# print(str(data), type(data))  # dict
# e_data = eval(str(data))
# print(e_data, type(e_data))  # dict

# # eval()会执行任何输入的代码，有点不安全倒是


# import json

# data = {'name': 'python', 'price': 45.6}
# # 序列化
# json_string = json.dumps(data)
# print(json_string, type(json_string))  # str

# # 反序列化
# restored_data = json.loads(json_string)
# print(restored_data, type(restored_data))  # dict

# import os
# import PIL.Image as Image
# img_path_a = r"D:\Ddesktop\ppt\work\luoshuan0516\trainval\images\trainval\MAX_2630.JPG"
# if Image.open(img_path_a):
#     print('a')
# img_path_b = r"D:\Ddesktop\ppt\work\luoshuan0516\trainval\images\trainval\MAX_2630.jpg"
# if Image.open(img_path_b):
#     print('b')