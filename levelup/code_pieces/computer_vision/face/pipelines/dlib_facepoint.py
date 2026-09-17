import cv2
import dlib
import argparse
import numpy as np
import os

def detect_face_landmarks(image_path, predictor_path, save_path=None):
    """
    检测图片中的人脸关键点并可视化
    
    Args:
        image_path: 输入图片路径
        predictor_path: dlib 68点人脸关键点预测器模型路径
        save_path: 结果保存路径（可选）
    
    Returns:
        None
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"预测器模型文件不存在: {predictor_path}")
    
    # 初始化dlib的人脸检测器和关键点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 转换为灰度图（关键点检测需要）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = detector(gray, 0)
    print(f"检测到 {len(faces)} 个人脸")
    
    if len(faces) == 0:
        print("未检测到人脸！")
        return
    
    # 对每个人脸进行关键点检测
    for i, face in enumerate(faces):
        # 获取68个关键点
        shape = predictor(gray, face)
        
        # 将关键点转换为numpy数组 (68, 2)
        landmarks = np.zeros((68, 2), dtype=int)
        for j in range(68):
            landmarks[j] = (shape.part(j).x, shape.part(j).y)
        
        # 绘制人脸框
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), 
                      (0, 255, 0), 2)
        
        # 绘制68个关键点
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        
        # 可选：标注关键点序号（便于查看）
        # for idx, (x, y) in enumerate(landmarks):
        #     cv2.putText(image, str(idx), (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    # 显示结果
    cv2.namedWindow("Face Landmarks", cv2.WINDOW_NORMAL)
    cv2.imshow("Face Landmarks", image)
    
    # 保存结果（如果指定了保存路径）
    if save_path:
        # 创建保存目录（如果不存在）
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        cv2.imwrite(save_path, image)
        print(f"结果已保存至: {save_path}")
    
    # 等待按键退出
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return landmarks  # 返回最后一个人脸的关键点坐标

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Dlib人脸关键点检测脚本')
    parser.add_argument('--image', '-i', required=True, help='输入图片路径（必填）')
    parser.add_argument('--predictor', '-p', 
                        default='shape_predictor_68_face_landmarks.dat',
                        help='dlib人脸关键点预测器模型路径（默认：shape_predictor_68_face_landmarks.dat）')
    parser.add_argument('--save', '-s', help='结果保存路径（可选，例如：result.jpg）')
    
    args = parser.parse_args()
    
    try:
        # 执行关键点检测
        landmarks = detect_face_landmarks(args.image, args.predictor, args.save)
        print(f"\n关键点检测完成！最后一个人脸的前5个关键点坐标：")
        for i in range(5):
            print(f"关键点 {i}: ({landmarks[i][0]}, {landmarks[i][1]})")
    
    except Exception as e:
        print(f"错误：{e}")
        print("\n使用提示：")
        print("1. 请确保已下载shape_predictor_68_face_landmarks.dat模型文件")
        print("   下载地址：http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("2. 解压后将模型文件放在脚本同目录，或通过-p参数指定路径")
        print("3. 示例命令：python face_landmarks.py -i test.jpg -p shape_predictor_68_face_landmarks.dat -s result.jpg")

if __name__ == "__main__":
    main()