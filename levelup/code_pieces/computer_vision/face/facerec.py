# 基于dlib的人脸识别工具库  face_recognition

import face_recognition 
from PIL import Image, ImageDraw
import time 

# 1. 加载图片和识别
ss = time.perf_counter()
imp = r"ims/a.jpg"
image_array = face_recognition.load_image_file(imp)
face_locations = face_recognition.face_locations(image_array)
face_landmarks_list = face_recognition.face_landmarks(image_array, face_locations)
tt = time.perf_counter()
print("time cost per img:", f"{1/(tt-ss) :.2f} fps")  # 2fps  

# 2. 将 numpy 数组转换为 PIL 图片对象，以便绘图
pil_image = Image.fromarray(image_array)
draw = ImageDraw.Draw(pil_image)

# 3. 画人脸框 (Bounding Box)
for (top, right, bottom, left) in face_locations:
    # ⚠️注意：face_recognition 返回的是 (top, right, bottom, left)
    # PIL 需要的是 [(left, top), (right, bottom)]
    draw.rectangle([ (left, top), (right, bottom) ], outline="red", width=3)

# 4. 画人脸特征点 (Landmarks)
# face_landmarks_list 是一个列表，里面每个元素是一个字典，对应一张脸
for face_landmarks in face_landmarks_list:
    # 遍历具体的特征（如下巴、眉毛、鼻子、眼睛、嘴唇）
    for feature_name, points in face_landmarks.items():
        # points 是一个坐标列表 [(x,y), (x,y)...]
        # 使用 line 方法把点连成线
        draw.line(points, fill="green", width=2)

# 5. 显示图片
pil_image.show() 
# 如果在 Jupyter 里，直接输入 pil_image 即可显示



# 人脸识别   ## 可以做