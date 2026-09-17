#  OS:  windows11 x64
#  mediapipe合适版本： 0.10.14、0.10.18、0.10.20、0.10.21   
## pip install mediapipe==0.10.21 protobuf opencv-python

## 直播模式  --代码

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

# === 2. 官方提供的可视化函数 (已整合) ===
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # 循环遍历检测到的人脸
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # 将 Task 的结果转换为 Protobuf 格式，以便用 solutions.drawing_utils 绘制
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in face_landmarks
        ])

        # 1. 绘制网格 (Tesselation)
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        
        # 2. 绘制轮廓 (Contours)
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        
        # 3. 绘制虹膜 (Irises)
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

# === 3. 核心：定义回调函数 ===
# 这是一个全局变量，用于在主线程和回调线程之间传递最新的推理结果
latest_result = None

def print_result(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """
    异步回调函数：当 MediaPipe 算完一帧后，会自动调用这个函数。
    注意：这个函数是在 MediaPipe 的线程中运行的，尽量不要在这里做耗时的绘图操作。
    """
    global latest_result
    latest_result = result

# === 4. 初始化模型 (LIVE_STREAM 模式) ===
model_path = r'ims/face_landmarker.task' # <--- 请确认路径
if not os.path.exists(model_path):
    print(f"找不到模型文件: {model_path}")
    sys.exit()

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM, # 【关键】设置为直播模式
    num_faces=1,
    result_callback=print_result # 【关键】注册回调函数
)

# === 5. 打开摄像头并循环 ===
# 0 通常是默认摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    sys.exit()

print("摄像头已启动。按 'q' 键退出，按 'ESC' 键退出。")

# 获取摄像头参数，确保录像文件尺寸匹配
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0 # 默认30帧


# save
output_filename = 'ims/facemesh_cam.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 编码器
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))


# 开始运行检测器
with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        # MediaPipe 需要 RGB，OpenCV 是 BGR
        # 注意：这里我们做个镜像翻转 (flip)，因为看镜子习惯是反的
        frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 获取当前时间戳 (毫秒)
        timestamp_ms = int(time.time() * 1000)
        
        # 【关键步骤】异步发送数据
        # 这里只是把图丢进去，不会阻塞代码运行
        landmarker.detect_async(mp_image, timestamp_ms)
        
        # === 绘制逻辑 ===
        # 检查有没有拿到最新的结果
        if latest_result is not None:
            # 使用官方提供的函数在当前帧上绘制
            # 注意：latest_result 可能是上一帧的计算结果（略有延迟），但在直播中肉眼难以察觉
            annotated_image = draw_landmarks_on_image(rgb_frame, latest_result)
            
            # 转回 BGR 用于 OpenCV 显示
            display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        else:
            # 如果还没算出结果，就直接显示原图
            display_image = frame

        # 写入 
        out.write(display_image)
        
        # 显示画面
        cv2.imshow('MediaPipe Live Face Mesh', display_image)

        # 退出检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # q 或 ESC
            break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"录制完成，文件已保存: {output_filename}")