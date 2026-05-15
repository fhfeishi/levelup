## pip install mediapipe==0.10.21 protobuf opencv-python
#  OS:  windows11 x64
#  mediapipe合适版本： 0.10.14、0.10.18、0.10.20、0.10.21   
## pip install mediapipe==0.10.21 protobuf opencv-python

import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# === 1. 环境与伪装类准备 ===
try:
    from mediapipe import solutions
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"环境报错: {e}")
    sys.exit()

mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_face_mesh = solutions.face_mesh

class FakeLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def HasField(self, field):
        return False

class FakeLandmarkList:
    def __init__(self):
        self.landmark = [] 

# === 2. 初始化模型 ===
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
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# === 3. 视频输入配置 ===
video_path = "ims/head2.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"无法打开视频: {video_path}")
    sys.exit()

# 获取原视频属性
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps == 0: fps = 30

# === 4. 视频输出配置 (新增部分) ===
output_path = "ims/output_demo.mp4"
# 左右拼接后，宽度变两倍，高度不变
output_size = (width * 2, height) 

# 初始化写入器 (mp4v 编码兼容性较好)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

print(f"开始处理... 结果将保存为: {output_path}")
print(f"输出分辨率: {output_size}")

# === 5. 处理循环 ===
# 屏幕显示时的缩放比例 (不影响保存的视频清晰度)
DISPLAY_SCALE = 0.6 

with FaceLandmarker.create_from_options(options) as landmarker:
    frame_index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("视频处理完成！")
            break

        # --- A. 推理 ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_timestamp_ms = int((frame_index * 1000) / fps)
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        # --- B. 绘制 ---
        annotated_image = frame.copy()
        
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                fake_proto = FakeLandmarkList()
                for landmark in face_landmarks:
                    fake_proto.landmark.append(FakeLandmark(landmark.x, landmark.y, landmark.z))

                # 绘制网格
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=fake_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # 绘制轮廓
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=fake_proto,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    
                # 绘制虹膜
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=fake_proto,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # --- C. 拼接与保存 ---
        
        # 添加水印
        cv2.putText(frame, "Original", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(annotated_image, "MediaPipe Mesh", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated_image, "Mesh", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 1. 左右拼接 (高画质版)
        combined_frame = cv2.hconcat([frame, annotated_image])

        # 2. 【关键】写入视频文件 (必须写入这个高画质的 combined_frame)
        out.write(combined_frame)

        # --- D. 屏幕显示 ---
        # 缩放一下再显示，不然屏幕放不下
        h, w_curr = combined_frame.shape[:2]
        display_frame = cv2.resize(combined_frame, (int(w_curr * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
        
        cv2.imshow('Processing...', display_frame)
        
        frame_index += 1
        
        # 按 'q' 提前结束
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户中断处理")
            break

# === 6. 资源释放 ===
cap.release()
out.release() # 这一步非常重要，否则视频文件会损坏
cv2.destroyAllWindows()
print(f"完成！视频已保存至当前目录: {output_path}")