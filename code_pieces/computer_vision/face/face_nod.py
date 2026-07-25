# 对人脸图片进行扭曲变换，让图片沿着不平行于图像平面的方向进行小角度扭曲、旋转  
# 第一步：模拟人点头的动作，将人脸图片沿着不平行于图像平面的方向进行小角度扭曲、旋转
# ## 透视变换（Perspective Transformation）
# ## 模拟“点头”动作（在 3D 空间中绕 X 轴旋转）,
# 不能简单地使用 2D 旋转（cv2.getRotationMatrix2D），因为那是绕 Z 轴旋转（即歪头）。


from pathlib import Path
import numpy as np 
import cv2 
import math 

# 计算 3D 旋转的透视变换矩阵
def get_perspective_transform_matrix(w, h, angle_x=0, angle_y=0, angle_z=0, scale=1.0, f_factor=1.0):
    """
    计算 3D 旋转的透视变换矩阵
    :param w: 图像宽度
    :param h: 图像高度
    :param angle_x: 绕 X 轴旋转角度（点头），单位：度
    :param angle_y: 绕 Y 轴旋转角度（摇头），单位：度
    :param angle_z: 绕 Z 轴旋转角度（平面旋转），单位：度
    :param scale: 缩放比例
    :param f_factor: 焦距因子，控制透视畸变程度。值越小畸变越明显（类似于广角），通常设为 1.0 左右
    :return: 3x3 透视变换矩阵
    """
    # 1. 角度转弧度
    rad_x = math.radians(angle_x)
    rad_y = math.radians(angle_y)
    rad_z = math.radians(angle_z)

    # 2. 定义焦距 (模拟相机的焦距，通常与图像尺寸相关)
    d = np.sqrt(h**2 + w**2)
    f = d * f_factor

    # 3. 将 2D 图像中心移动到坐标原点 (0,0,0)
    # 图像原本在 Z=0 平面上
    # 原始的四个角点
    pts_src = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    # 4. 构建旋转矩阵 (RX, RY, RZ)
    # 绕 X 轴旋转 (点头)
    RX = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rad_x), -np.sin(rad_x), 0],
        [0, np.sin(rad_x), np.cos(rad_x), 0],
        [0, 0, 0, 1]
    ])
    
    # 绕 Y 轴旋转 (摇头 - 这里虽然没用到，但为了通用性保留)
    RY = np.array([
        [np.cos(rad_y), 0, np.sin(rad_y), 0],
        [0, 1, 0, 0],
        [-np.sin(rad_y), 0, np.cos(rad_y), 0],
        [0, 0, 0, 1]
    ])

    # 绕 Z 轴旋转 (平面旋转)
    RZ = np.array([
        [np.cos(rad_z), -np.sin(rad_z), 0, 0],
        [np.sin(rad_z), np.cos(rad_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # 组合旋转矩阵 R = RX * RY * RZ
    R = RX @ RY @ RZ

    # 变换矩阵 T (平移中心 -> 旋转 -> 平移回中心)
    # 这里的逻辑是：
    # 1. (x - w/2, y - h/2, 0, 1) -> 移到中心
    # 2. Apply Rotation
    # 3. Apply Translation (0, 0, f) -> 将物体推远，放在相机前方 f 处以便投影
    
    # 为了简化计算，我们直接对四个角点进行操作
    
    dst_pts = []
    for pt in pts_src:
        # 将点转为 3D 坐标，并移动中心到原点
        x, y = pt[0] - w/2, pt[1] - h/2
        z = 0
        vec = np.array([x, y, z, 1])
        
        # 旋转
        vec_rot = R @ vec
        
        # 投影回 2D 平面
        # 投影公式: x' = x * (f / (f + z)), y' = y * (f / (f + z))
        # 注意：这里我们假设相机在 (0,0,-f)，物体被放到了 z=0 附近。
        # 或者更简单的透视除法：x_proj = x_rot * f / (z_rot + f) + w/2
        
        x_rot, y_rot, z_rot = vec_rot[0], vec_rot[1], vec_rot[2]
        
        # 防止除以零
        div = f + z_rot if (f + z_rot) != 0 else 0.0001
        
        x_proj = (x_rot * f / div) + w/2
        y_proj = (y_rot * f / div) + h/2
        
        dst_pts.append([x_proj, y_proj])

    dst_pts = np.array(dst_pts, dtype=np.float32)

    # 5. 利用源点和目标点计算透视变换矩阵 Homography
    M = cv2.getPerspectiveTransform(pts_src, dst_pts)
    return M

# 执行点头变换
def nod_trans(image_path, output_dir, angle=15):
    """
    执行点头变换
    :param image_path: 图片路径
    :param output_dir: 输出目录
    :param angle: 点头角度 (正数向下看，负数向上看)
    """
    if not image_path:
        print("未找到图片")
        return

    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    h, w = img.shape[:2]
    
    # 获取变换矩阵
    # angle_x 控制点头动作
    M = get_perspective_transform_matrix(w, h, angle_x=angle, f_factor=1.0)
    
    # 应用透视变换
    # borderValue 定义背景填充色，这里设为白色(255,255,255)或黑色(0,0,0)
    warped_img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # 保存结果
    file_name = Path(image_path).name
    save_path = Path(output_dir) / f"nod_{angle}_{file_name}"
    cv2.imwrite(str(save_path), warped_img)
    print(f"已保存: {save_path}")

# 执行歪头变换 (Z轴旋转)
def tilt_trans(image_path, output_dir, angle=5):
    """
    执行歪头变换 (Z轴旋转)
    :param angle: 歪头角度 (正数向右歪，负数向左歪)
    """
    if not image_path: return

    img = cv2.imread(str(image_path))
    if img is None: return

    h, w = img.shape[:2]
    
    # 核心修改：使用 angle_z 控制歪头
    # 同时为了增加"生动感"，我们稍微加一点点 scale (1.02)，模拟呼吸时的轻微前倾
    M = get_perspective_transform_matrix(w, h, angle_x=0, angle_y=0, angle_z=angle, scale=1.02)
    
    warped_img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    file_name = Path(image_path).name
    # 命名区分：tilt 代表歪头
    save_path = Path(output_dir) / f"tilt_{angle}_{file_name}"
    cv2.imwrite(str(save_path), warped_img)
    print(f"已保存歪头效果: {save_path}")


# [高级] 模拟随机的细微动作（混合点头、摇头、歪头）
def random_liveness_trans(image_path, output_dir, count=3, xyz=(3,0,2), random_=False):
    """
    [高级] 模拟随机的细微动作（混合点头x、摇头y、歪头z）
    生成多张看似静止但有细微差别的图片，用于合成 GIF 或数据集增强
    """
    import random
    
    if not image_path: return
    img = cv2.imread(str(image_path))
    if img is None: return
    h, w = img.shape[:2]

    file_name = Path(image_path).stem
    suffix = Path(image_path).suffix

    if random_:
        for i in range(count):
            # 生成细微的随机角度 (-3度 到 3度 之间)
            # 这种微小的混合角度最能模拟"活体"的感觉
            x_, y_, z_ = abs(xyz[0]), abs(xyz[1]), abs(xyz[2])
            ax = random.uniform(-x_, x_)  # 微点头
            ay = random.uniform(-y_, y_)  # 微摇头
            az = random.uniform(-z_, z_)  # 微歪头
            
            M = get_perspective_transform_matrix(w, h, angle_x=ax, angle_y=ay, angle_z=az, scale=1.0)
            
            warped_img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            
            save_path = Path(output_dir) / f"{file_name}_{ax:.1f}_{int(ay)}_{az:.1f}_{i}{suffix}"
            cv2.imwrite(str(save_path), warped_img)
            print(f"已保存微动作: {save_path} (X:{ax:.1f}, Y:{ay:.1f}, Z:{az:.1f})")
    else:
        # 固定角度
        ax, ay, az = xyz[0], xyz[1], xyz[2]
        M = get_perspective_transform_matrix(w, h, angle_x=ax, angle_y=ay, angle_z=az, scale=1.0)
        warped_img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        save_path = Path(output_dir) / f"{file_name}_{ax:.1f}_{int(ay)}_{az:.1f}{suffix}"
        cv2.imwrite(str(save_path), warped_img)
        print(f"已保存微动作: {save_path} (X:{ax:.1f}, Y:{ay:.1f}, Z:{az:.1f})")
        
        
if __name__ == "__main__":
    # 示例使用
    srcim_dir = r"./ims/nod"
    dstim_dir = r"./ims/nod_out"
    Path(dstim_dir).mkdir(parents=True, exist_ok=True) # parents=True 以防父目录不存在

    im_path = next(Path(srcim_dir).glob('*.png'), None)
    
    # # 模拟点头（向下看20度）
    # nod_trans(im_path, dstim_dir, angle=20)

    # # 模拟抬头（向上看20度）
    # nod_trans(im_path, dstim_dir, angle=-20)
    
    
    # XYZ: 点头、摇头、歪头
    random_liveness_trans(im_path, dstim_dir, count=10, xyz=(15,0,7), random_=False)


