import dlib
import cv2
import numpy as np
from pathlib import Path
import time 

# numpy         1.26.4    # 1.x
# opencv-python 4.9.0.80  # cv版本不能太新了
# dlib          19.22.99  # python3.10  .whl

def create_dlib_compatible_image(cv_image):
    """
    将OpenCV图像转换为dlib 100%兼容的格式
    
    关键点：
    1. 确保uint8
    2. 确保灰度图是2D
    3. 确保内存连续（C-contiguous）
    4. 不使用dlib的图像加载函数
    """
    # 转换为灰度图
    if len(cv_image.shape) == 3:
        if cv_image.shape[2] == 4:  # RGBA
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY)
        else:  # BGR
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    elif len(cv_image.shape) == 2:
        gray = cv_image
    else:
        raise ValueError(f"不支持的图像维度: {cv_image.shape}")
    
    # 强制uint8
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    # 确保2D（去除多余的单通道维度）
    gray = np.squeeze(gray)
    
    # 最关键：确保内存连续
    # 即使前面的操作看起来没问题，也要强制复制确保连续性
    gray = np.ascontiguousarray(gray)
    
    # 验证
    assert gray.dtype == np.uint8, f"dtype错误: {gray.dtype}"
    assert len(gray.shape) == 2, f"维度错误: {gray.shape}"
    assert gray.flags['C_CONTIGUOUS'], "内存不连续"
    
    return gray


def main():
    # 初始化检测器
    detector = dlib.get_frontal_face_detector()
    
    # 图像路径
    image_path = r"D:\code\robotss\u2net\train_data\robot_data\images\frame_000030.jpg"
    
    # 检查文件
    if not Path(image_path).exists():
        print(f"❌ 文件不存在: {image_path}")
        return
    
    print(f"正在读取: {image_path}")
    
    # 方案1：用OpenCV读取（推荐）
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"❌ OpenCV无法读取图片")
        return
    
    print(f"✓ 图像读取成功: {image.shape}")
    
    # 转换为dlib兼容格式
    try:
        gray = create_dlib_compatible_image(image)
        print(f"✓ 格式转换成功: shape={gray.shape}, dtype={gray.dtype}")
        print(f"  内存连续: {gray.flags['C_CONTIGUOUS']}")
        print(f"  像素范围: [{gray.min()}, {gray.max()}]")
    except Exception as e:
        print(f"❌ 格式转换失败: {e}")
        return
    
    # 人脸检测
    print("\n开始人脸检测...")
    start_time = time.perf_counter()
    
    try:
        # 重要：传入numpy数组，不使用dlib的加载函数
        faces = detector(gray, 1)  # upsample=1
        
        elapsed = time.perf_counter() - start_time
        print(f"✓ 检测完成")
        print(f"  检测到人脸: {len(faces)}")
        print(f"  耗时: {elapsed:.3f}秒")
        
    except RuntimeError as e:
        print(f"❌ 检测失败: {e}")
        print(f"\n调试信息:")
        print(f"  dlib版本: {dlib.__version__}")
        print(f"  numpy版本: {np.__version__}")
        print(f"  OpenCV版本: {cv2.__version__}")
        return
    
    # 如果没检测到人脸
    if len(faces) == 0:
        print("\n⚠️ 未检测到人脸，尝试调整参数...")
        # 尝试不上采样（更快但可能漏检）
        faces = detector(gray, 0)
        print(f"  不上采样结果: {len(faces)} 张人脸")
    
    # 绘制结果
    for i, face in enumerate(faces):
        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()
        
        print(f"\n人脸 {i+1}:")
        print(f"  位置: x={x}, y={y}")
        print(f"  尺寸: w={w}, h={h}")
        
        # 画框
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"Face {i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 保存结果
    output_path = "dlib_result.jpg"
    cv2.imwrite(output_path, image)
    print(f"\n✓ 结果已保存: {output_path}")
    
    # 显示
    cv2.namedWindow("Dlib Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Dlib Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()