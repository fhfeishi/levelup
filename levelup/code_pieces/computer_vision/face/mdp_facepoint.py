#  OS:  windows11 x64
#  mediapipe合适版本： 0.10.14、0.10.18、0.10.20、0.10.21   
## pip install mediapipe==0.10.21 protobuf opencv-python

import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import time

# === 1. 导入工具 ===
try:
    from mediapipe import solutions
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"环境报错: {e}")
    sys.exit()

# === 2. 准备绘图工具 ===
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_face_mesh = solutions.face_mesh

# === 3. 【核心魔法修正】升级伪装类 ===
class FakeLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    # === 关键修复：添加 Protobuf 协议方法 ===
    # drawing_utils 会调用这个方法检查 'visibility' 或 'presence'
    # 我们直接返回 False，让它认为这些字段不存在，从而强制绘制所有点
    def HasField(self, field):
        return False

class FakeLandmarkList:
    def __init__(self):
        self.landmark = [] 

# === 4. 初始化模型 ===
model_path = r'ims/face_landmarker.task' 
if not os.path.exists(model_path):
    print(f"找不到模型文件: {model_path}")
    sys.exit()

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# === 5. 推理 ===
image_path = "ims/b.jpg" # <--- 确认路径
cv_mat = cv2.imread(image_path)
if cv_mat is None:
    print("无法读取图片")
    sys.exit()

ss = time.perf_counter()
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cv_mat, cv2.COLOR_BGR2RGB))

with FaceLandmarker.create_from_options(options) as landmarker:
    detection_result = landmarker.detect(mp_image)
tt1 = time.perf_counter()
print("time cost per image :det:", f"{1/(tt1 -ss):.2f}")

# === 6. 可视化 (使用升级后的伪装数据) ===
annotated_image = cv_mat.copy()

if detection_result.face_landmarks:
    print(f"检测到人脸，开始绘制...")
    
    for face_landmarks in detection_result.face_landmarks:
        
        # A. 数据转换
        fake_proto = FakeLandmarkList()
        for landmark in face_landmarks:
            fake_proto.landmark.append(FakeLandmark(landmark.x, landmark.y, landmark.z))

        # B. 绘图 (现在不会报错了)
        # 1. 画网格 (Tesselation)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=fake_proto,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # 2. 画轮廓 (Contours)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=fake_proto,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
        # 3. 画虹膜 (Irises)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=fake_proto,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    cv2.imshow("MediaPipe Mesh", annotated_image)
    tt2 = time.perf_counter()
    print("time cost per image :show:", f"{1/(tt2 -ss):.2f}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未检测到人脸")