#  OS:  windows11 x64
#  mediapipe合适版本： 0.10.14、0.10.18、0.10.20、0.10.21   
## pip install mediapipe==0.10.21 protobuf opencv-python


# face det

import cv2
import mediapipe as mp
import numpy as np

# 显式导入 tasks 组件，避免 import 错误
from mediapipe.tasks.python import vision

# === 配置路径 ===
# 请确保该文件就在你的 ims 文件夹下
model_path = r"ims/blaze_face_short_range.tflite"
# image_path = "ims/a.jpg"  # 你的测试图片路径
image_path = "ims/nod_out/20260108_153541_284_15.0_0_7.0.png"  # 你的测试图片路径

scale_g = 0.1

# === 1. 初始化 FaceDetector (专用于 .tflite 模型) ===
BaseOptions = mp.tasks.BaseOptions
FaceDetector = vision.FaceDetector
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

# 创建检测器选项
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.5
)

# === 2. 执行推理 ===
try:
    with FaceDetector.create_from_options(options) as detector:
        # 读取图片
        cv_mat = cv2.imread(image_path)
        img_h,img_w = cv_mat.shape[:2]
        if cv_mat is None:
            raise ValueError(f"无法找到图片: {image_path}")
            
        # 转换图片格式 (OpenCV BGR -> MediaPipe Image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cv_mat, cv2.COLOR_BGR2RGB))
        
        # 检测
        detection_result = detector.detect(mp_image)
        
        # === 3. 绘制结果 (6个关键点 + 框) ===
        annotated_image = cv_mat.copy()
        
        if detection_result.detections:
            print(f"检测到 {len(detection_result.detections)} 张人脸")
            for detection in detection_result.detections:
                # 画框
                bbox = detection.bounding_box
                # start_point = (bbox.origin_x, bbox.origin_y)
                # end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height *1.1)
                # cv2.rectangle(annotated_image, start_point, end_point, (0, 255, 0), 2)
                
                # 1. 定义扩充比例 (10% = 0.1)
                scale = scale_g   # 0.1
                
                # 2. 计算需要向四周延伸的像素量 (宽高的 10%，分摊到左右/上下各一半)
                # 比如宽扩充 10%，那么左边扩 5%，右边扩 5%
                pad_w = int(bbox.width * scale / 2)
                pad_h = int(bbox.height * scale / 2)
                
                # 3. 计算新的坐标 (原坐标 - 偏移量)
                new_x1 = bbox.origin_x - pad_w
                new_y1 = bbox.origin_y - pad_h
                new_x2 = bbox.origin_x + bbox.width + pad_w
                new_y2 = bbox.origin_y + bbox.height + pad_h
                
                # 4. 边界检查 (Clamp) - 这一步非常重要，防止报错
                # 确保不小于0，不超过图片宽高
                new_x1 = max(0, new_x1)
                new_y1 = max(0, new_y1)
                new_x2 = min(img_w, new_x2)
                new_y2 = min(img_h, new_y2)
                
                # 5. 画框
                cv2.rectangle(annotated_image, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
                
                # 画关键点 (BlazeFace 只有6个关键点：眼、鼻、嘴、耳)
                # # 注意：这个模型没有 468 个点
                # if detection.keypoints:
                #     for kp in detection.keypoints:
                #         x = int(kp.x * cv_mat.shape[1])
                #         y = int(kp.y * cv_mat.shape[0])
                #         cv2.circle(annotated_image, (x, y), 4, (0, 0, 255), -1)
        else:
            print("未检测到人脸")

        # 显示
        cv2.imshow("Result", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except Exception as e:
    print(f"发生错误: {e}")
    # 如果报错 RuntimeError: Unable to open file... 请检查 model_path 是否绝对正确