import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 去掉mask中的白色干扰区域，初步处理、处理明显的噪点

def process_mask(image_path, region1, region2, denoise=True):
    """
    处理mask图片：去噪 + 清除指定区域
    
    Args:
        image_path: 输入图片路径
        region1: 第一个区域 ((x1, y1), (x2, y2))
        region2: 第二个区域 ((x1, y1), (x2, y2))
        denoise: 是否去除小噪点
    """
    # 读取图片（灰度）
    mask = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Warning: Cannot read {image_path}")
        return None
    
    # # 1. 可选：去除噪点（形态学操作）
    # if denoise:
    #     # 开运算：先腐蚀后膨胀，去除小白点
    #     kernel = np.ones((3, 3), np.uint8)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
    #     # 可选：闭运算，填充小黑洞
    #     # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
    #     # 可选：去除小连通区域（面积阈值）
    #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    #     min_area = 50  # 小于50像素的区域视为噪点
    #     for i in range(1, num_labels):  # 跳过背景（label=0）
    #         if stats[i, cv2.CC_STAT_AREA] < min_area:
    #             mask[labels == i] = 0
    
    # 2. 清除指定区域（设为黑色背景）
    (x1, y1), (x2, y2) = region1
    mask[y1:y2, x1:x2] = 0
    
    (x1, y1), (x2, y2) = region2
    mask[y1:y2, x1:x2] = 0
    
    return mask

def batch_process_masks(input_dir, output_dir=None, region1=None, region2=None):
    """
    批量处理mask图片
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出目录（默认与输入相同）
        region1, region2: 要清除的区域
    """
    input_path = Path(input_dir)
    
    # 如果未指定输出目录，在原目录保存
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 默认区域
    if region1 is None:
        region1 = ((240, 180), (310, 230))
    if region2 is None:
        region2 = ((290, 370), (370, 450))
    
    # 获取所有png图片
    image_files = list(input_path.glob('*.png'))
    
    if not image_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Region 1: {region1}")
    print(f"Region 2: {region2}")
    print("-" * 50)
    
    # 批量处理
    success_count = 0
    for img_path in tqdm(image_files, desc="Processing"):
        # 处理mask
        processed_mask = process_mask(img_path, region1, region2, denoise=True)
        
        if processed_mask is None:
            continue
        
        # 生成新文件名：frame -> mask
        original_name = img_path.stem  # 不带后缀的文件名
        new_name = original_name+ '_mask'
        
        # # 如果文件名没有frame，则添加mask_前缀
        # if 'frame' not in original_name:
        #     new_name = f'mask_{original_name}'
        
        # 保存
        output_file = output_path / f'{new_name}.png'
        cv2.imwrite(str(output_file), processed_mask)
        success_count += 1
    
    print(f"\n✓ Successfully processed {success_count}/{len(image_files)} images")
    print(f"Output directory: {output_path}")

# ============ 使用示例 ============
if __name__ == '__main__':
    # 配置路径
    INPUT_DIR = r'D:\ddesktop\robotss\train-label-1467\train-label-xiangdong1164'  # 你的输入目录
    OUTPUT_DIR = None  # None表示在原目录保存，或指定新目录如 'path/to/output'
    
    # 定义要清除的区域（你的坐标）
    REGION_1 = ((240, 180), (310, 230))  # (x1, y1) -> (x2, y2)
    REGION_2 = ((290, 370), (370, 450))
    
    # 执行批量处理
    batch_process_masks(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        region1=REGION_1,
        region2=REGION_2
    )