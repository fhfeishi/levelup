#  OS:  windows11 x64
#  mediapipe合适版本： 0.10.14、0.10.18、0.10.20、0.10.21   
## pip install mediapipe==0.10.21 protobuf opencv-python

import cv2
import mediapipe as mp
import numpy as np
import sys
import os

mediapipe_version = str(mp.__version__)
# ==========================================================
# 1. 环境检查与导入
# ==========================================================
try:
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"环境有问题，请确认 mediapipe protobuf numpy opencv-python的版本")
    print(f"具体错误: {e}")
    sys.exit()

# 绘图工具初始化
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_face_mesh = solutions.face_mesh

# ==========================================================
# 2. 初始化模型 (Tasks API)
# ==========================================================
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

# ==========================================================
# 3. 视频输入与输出配置
# ==========================================================
video_path = "ims/xx.mp4" # <--- 输入视频
output_path = "ims/xx_result_with_mesh.mp4" # <--- 输出结果

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"无法打开视频: {video_path}")
    sys.exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if fps == 0: fps = 30

# 我们做一个左右对比视频，所以宽度 x2
out_size = (width * 2, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, out_size)

print(f"开始处理... 结果将保存为: {output_path}")

# ==========================================================
# 4. 循环处理
# ==========================================================
with FaceLandmarker.create_from_options(options) as landmarker:
    frame_index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # A. 转换与推理
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int((frame_index * 1000) / fps)
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # B. 绘制 (在 0.10.14 版本下，使用官方转换方法)
        annotated_image = frame.copy()
        
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                
                # 【关键】将 Tasks API 的输出转换为 Protobuf 格式
                # 这样 drawing_utils 就能直接画了，不需要自己写伪装类
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                    for lm in face_landmarks
                ])

                # 1. 画网格
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # 2. 画轮廓
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                # 3. 画虹膜
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # C. 拼接画面
        # 添加文字
        cv2.putText(frame, "Original", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"MediaPipe {mediapipe_version}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        combined = cv2.hconcat([frame, annotated_image])
        
        # D. 写入文件
        out.write(combined)
        
        # E. 屏幕预览 (缩小一点以免撑爆屏幕)
        preview = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Processing...', preview)
        
        frame_index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
print("完成！")