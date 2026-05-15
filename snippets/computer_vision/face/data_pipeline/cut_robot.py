import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_bbox_dict(bbox_txt):
    """加载bbox.txt"""
    bboxes = {}
    with open(bbox_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                stem = parts[0]
                coords = list(map(int, parts[1].split(',')))
                if len(coords) == 4:
                    bboxes[stem] = coords  # [x, y, w, h]
    return bboxes

def crop_fixed_size(image, x, y, w, h, crop_size=1080):
    """
    以bbox为中心裁剪固定尺寸的区域
    
    Args:
        image: 输入图像
        x, y, w, h: bbox坐标
        crop_size: 裁剪尺寸（正方形）
    """
    img_h, img_w = image.shape[:2]
    
    # 计算bbox中心点
    center_x = x + w // 2
    center_y = y + h // 2
    
    # 以中心点为基准，计算裁剪区域的左上角
    crop_x = center_x - crop_size // 2
    crop_y = center_y - crop_size // 2
    
    # 确保裁剪区域不越界
    if crop_x < 0:
        crop_x = 0
    if crop_y < 0:
        crop_y = 0
    if crop_x + crop_size > img_w:
        crop_x = img_w - crop_size
    if crop_y + crop_size > img_h:
        crop_y = img_h - crop_size
    
    # 裁剪
    cropped = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    
    return cropped

def batch_crop(jpg_dir, mask_dir, save_dir, bbox_txt, crop_size=1080):
    """批量裁剪jpg和mask为固定尺寸"""
    
    jpg_path = Path(jpg_dir)
    mask_path = Path(mask_dir)
    save_path = Path(save_dir)
    
    # 创建输出目录
    (save_path / 'images').mkdir(parents=True, exist_ok=True)
    (save_path / 'masks').mkdir(parents=True, exist_ok=True)
    
    # 加载bbox
    bboxes = load_bbox_dict(bbox_txt)
    print(f"✓ Loaded {len(bboxes)} bboxes from {bbox_txt}")
    
    # 获取所有jpg文件
    jpg_files = list(jpg_path.glob('*.jpg'))
    
    if not jpg_files:
        print(f"❌ No jpg files found in {jpg_dir}")
        return
    
    print(f"Found {len(jpg_files)} jpg files")
    print(f"Crop size: {crop_size}x{crop_size}")
    print(f"Output: {save_path}")
    print("="*60)
    
    success = 0
    no_bbox = 0
    no_mask = 0
    size_error = 0
    
    for jpg_file in tqdm(jpg_files, desc="Cropping"):
        stem = jpg_file.stem
        
        # 检查是否有bbox
        if stem not in bboxes:
            no_bbox += 1
            continue
        
        # 读取jpg
        image = cv2.imread(str(jpg_file))
        if image is None:
            continue
        
        # 检查图像尺寸是否足够
        if image.shape[0] < crop_size or image.shape[1] < crop_size:
            size_error += 1
            continue
        
        # 读取对应的mask
        mask_file = mask_path / f"{stem}.png"
        if not mask_file.exists():
            no_mask += 1
            continue
        
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            no_mask += 1
            continue
        
        # 获取bbox
        x, y, w, h = bboxes[stem]
        
        # 裁剪固定尺寸区域
        cropped_img = crop_fixed_size(image, x, y, w, h, crop_size)
        cropped_mask = crop_fixed_size(mask, x, y, w, h, crop_size)
        
        # 验证裁剪后的尺寸
        if cropped_img.shape[0] != crop_size or cropped_img.shape[1] != crop_size:
            size_error += 1
            continue
        
        # 保存
        cv2.imwrite(str(save_path / 'images' / jpg_file.name), cropped_img)
        cv2.imwrite(str(save_path / 'masks' / f"{stem}.png"), cropped_mask)
        
        success += 1
    
    # 统计
    print("\n" + "="*60)
    print(f"✅ Processing completed!")
    print(f"   Successfully cropped: {success}")
    print(f"   No bbox found: {no_bbox}")
    print(f"   No mask found: {no_mask}")
    print(f"   Image too small: {size_error}")
    print(f"\n💾 Output:")
    print(f"   Images: {save_path / 'images'}")
    print(f"   Masks:  {save_path / 'masks'}")
    print(f"   All images are {crop_size}x{crop_size}")
    print("="*60)

# ============ 运行 ============
if __name__ == '__main__':
    batch_crop(
        jpg_dir=r'D:\ddesktop\robotss\train-label-1467\train-data-robot303\train_data',
        mask_dir=r'D:\ddesktop\robotss\train-label-1467\train-data-robot303\train_cleaned',
        save_dir=r'D:\ddesktop\robotss\train-label-1467\train-data-robot303\im_mask',
        bbox_txt=r'D:\ddesktop\robotss\codespace\bbox.txt',
        crop_size=1080
    )