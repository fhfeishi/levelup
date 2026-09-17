# 文件名: det_facePoint.py
# 适配 MediaPipe 0.10+

import cv2
import numpy as np
from pathlib import Path
import json
import time
import urllib.request

# MediaPipe 0.10+ 新API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeFaceLandmarks:
    """MediaPipe人脸关键点检测（适配0.10+版本）"""
    
    def __init__(self):
        print("正在初始化MediaPipe Face Landmarker...")
        
        # 模型文件路径
        model_path = 'face_landmarker.task'
        
        # 如果模型不存在，自动下载
        if not Path(model_path).exists():
            print("正在下载模型文件...")
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✓ 模型已下载: {model_path}")
            except Exception as e:
                print(f"❌ 模型下载失败: {e}")
                print("请手动下载模型:")
                print(f"  URL: {url}")
                print(f"  保存到: {Path(model_path).absolute()}")
                raise
        
        # 创建检测器选项
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        
        # 创建检测器
        self.detector = vision.FaceLandmarker.create_from_options(options)
        print("✓ 初始化完成")
    
    def detect(self, image_path):
        """检测人脸关键点"""
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None
        
        h, w = image.shape[:2]
        
        # 转RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建MediaPipe Image对象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # 检测
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None, image
        
        # 获取第一个人脸的关键点（478个点）
        face_landmarks = detection_result.face_landmarks[0]
        
        # 转换为像素坐标
        landmarks = []
        for landmark in face_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        return landmarks, image
    
    def visualize(self, image, landmarks, show_indices=False):
        """可视化关键点"""
        vis_image = image.copy()
        
        # 画所有关键点
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_image, (x, y), 1, (0, 255, 0), -1)
            
            if show_indices and i % 20 == 0:
                cv2.putText(vis_image, str(i), (x+2, y-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # 画连接线
        self._draw_connections(vis_image, landmarks)
        
        return vis_image
    
    def _draw_connections(self, image, landmarks):
        """绘制面部轮廓连接线"""
        # 左眼
        left_eye = [33, 160, 158, 133, 153, 144, 145, 163, 33]
        self._draw_line(image, landmarks, left_eye, (0, 255, 255))
        
        # 右眼
        right_eye = [362, 385, 387, 263, 373, 380, 381, 382, 362]
        self._draw_line(image, landmarks, right_eye, (0, 255, 255))
        
        # 嘴巴外轮廓
        mouth_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 
                       375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
        self._draw_line(image, landmarks, mouth_outer, (255, 0, 255))
        
        # 脸部轮廓
        face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
        self._draw_line(image, landmarks, face_oval, (0, 200, 0))
    
    def _draw_line(self, image, landmarks, indices, color):
        """绘制连接线"""
        for i in range(len(indices) - 1):
            if indices[i] < len(landmarks) and indices[i+1] < len(landmarks):
                pt1 = landmarks[indices[i]]
                pt2 = landmarks[indices[i + 1]]
                cv2.line(image, pt1, pt2, color, 1)
    
    def save_landmarks(self, landmarks, output_path):
        """保存关键点坐标到JSON"""
        data = {
            'landmarks': landmarks,
            'num_points': len(landmarks)
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    """单张图片推理"""
    print("="*60)
    print("MediaPipe 人脸关键点检测 (v0.10+)")
    print("="*60)
    
    # 初始化
    try:
        detector = MediaPipeFaceLandmarks()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 图片路径
    image_path = r"D:\code\robotss\u2net\train_data\robot_data\images\frame_000030.jpg"
    
    if not Path(image_path).exists():
        print(f"❌ 文件不存在: {image_path}")
        return
    
    print(f"\n正在处理: {image_path}")
    
    # 检测
    start = time.perf_counter()
    landmarks, image = detector.detect(image_path)
    elapsed = time.perf_counter() - start
    
    print(f"✓ 推理耗时: {elapsed:.3f}秒")
    
    if landmarks is None:
        print("❌ 未检测到人脸")
        return
    
    print(f"✓ 检测到 {len(landmarks)} 个关键点")
    
    # 可视化
    vis_image = detector.visualize(image, landmarks, show_indices=False)
    
    # 保存结果
    output_image = "landmarks_result.jpg"
    cv2.imwrite(output_image, vis_image)
    print(f"✓ 可视化结果已保存: {output_image}")
    
    # 保存关键点坐标
    output_json = "landmarks.json"
    detector.save_landmarks(landmarks, output_json)
    print(f"✓ 关键点坐标已保存: {output_json}")
    
    # 打印前10个关键点
    print(f"\n前10个关键点:")
    for i, (x, y) in enumerate(landmarks[:10]):
        print(f"  点{i}: ({x}, {y})")
    
    # 显示
    print("\n按任意键关闭窗口...")
    cv2.namedWindow("Landmarks", cv2.WINDOW_NORMAL)
    cv2.imshow("Landmarks", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n✅ 完成！")


def batch_inference(input_dir, output_dir):
    """批量推理"""
    from tqdm import tqdm
    
    detector = MediaPipeFaceLandmarks()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(input_path.glob(ext))
    
    print(f"Found {len(image_files)} images")
    
    success = 0
    failed = []
    
    for img_file in tqdm(image_files, desc="Processing"):
        landmarks, image = detector.detect(img_file)
        
        if landmarks:
            # 保存可视化
            vis_image = detector.visualize(image, landmarks)
            cv2.imwrite(str(output_path / img_file.name), vis_image)
            
            # 保存坐标
            json_path = output_path / f"{img_file.stem}.json"
            detector.save_landmarks(landmarks, json_path)
            
            success += 1
        else:
            failed.append(img_file.name)
    
    print(f"\n✅ Success: {success}/{len(image_files)}")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for name in failed[:10]:
            print(f"   {name}")


if __name__ == "__main__":
    try:
        # 单张图片推理
        main()
        
        # 批量推理（取消注释使用）
        # batch_inference(
        #     input_dir=r"D:\code\robotss\u2net\train_data\robot_data\images",
        #     output_dir=r"D:\code\robotss\u2net\output_landmarks"
        # )
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()