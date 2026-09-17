import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os

# === 1. 环境导入 ===
try:
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"环境报错: {e}")
    sys.exit()

# 绘图工具引用
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_face_mesh = solutions.face_mesh

# === 2. 官方提供的可视化函数 ===
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

# === 3. 回调函数 ===
latest_result = None

def print_result(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

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
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=print_result
)

# === 5. 打开摄像头并循环 ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    sys.exit()

# ==========================================
# 【关键修改】设置隔帧参数
# ==========================================
frame_interval = 3  # 间隔2帧处理1帧 (即每2帧推理一次)
assert frame_interval > 0

frame_count = 0     # 帧计数器

print(f"摄像头已启动。隔帧处理模式: 每 {frame_interval} 帧推理一次。")

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 镜像翻转 & 颜色转换
        frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 帧计数自增
        frame_count += 1
        
        # ==========================================
        # 【关键修改】隔帧推理逻辑
        # ==========================================
        # 只有当计数器能被间隔整除时，才发送给 MediaPipe 进行推理
        if frame_count % frame_interval == 0:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)
        
        # ==========================================
        # 【关键修改】绘制逻辑
        # ==========================================
        # 注意：这里我们在“每一帧”都进行绘制，而不仅仅是推理的那一帧。
        # 如果当前帧被跳过了没推理，latest_result 会保持“上一次推理的结果”。
        # 这样画面会非常流畅，网格会紧贴人脸，不会出现闪烁。
        if latest_result is not None:
            annotated_image = draw_landmarks_on_image(rgb_frame, latest_result)
            display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        else:
            display_image = frame

        cv2.imshow('MediaPipe Live Face Mesh (Frame Skipping)', display_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()