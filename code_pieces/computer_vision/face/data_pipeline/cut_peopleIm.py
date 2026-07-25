# y: 60-780

import cv2
from pathlib import Path
from tqdm import tqdm

def crop_fixed_region(image, x, y, w, h):
    """按固定区域裁剪"""
    return image[y:y+h, x:x+w]

def batch_crop_fixed(jpg_dir, save_dir, crop_x=0, crop_y=60, crop_w=720, crop_h=720):
    """
    批量裁剪固定区域
    
    Args:
        jpg_dir: jpg和png所在目录
        save_dir: 输出目录
        crop_x, crop_y, crop_w, crop_h: 裁剪区域
    """
    data_path = Path(jpg_dir)
    save_path = Path(save_dir)
    
    # 创建输出目录
    (save_path / 'images').mkdir(parents=True, exist_ok=True)
    (save_path / 'masks').mkdir(parents=True, exist_ok=True)
    
    # 获取所有jpg文件
    jpg_files = list(data_path.glob('*.jpg'))
    
    if not jpg_files:
        print(f"❌ No jpg files found in {jpg_dir}")
        return
    
    print(f"Found {len(jpg_files)} jpg files")
    print(f"Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
    print(f"Output size: {crop_w}x{crop_h}")
    print(f"Output: {save_path}")
    print("="*60)
    
    success = 0
    no_mask = 0
    size_error = 0
    
    for jpg_file in tqdm(jpg_files, desc="Cropping"):
        stem = jpg_file.stem
        
        # 读取jpg
        image = cv2.imread(str(jpg_file))
        if image is None:
            continue
        
        # 检查图像尺寸
        img_h, img_w = image.shape[:2]
        if img_w < crop_x + crop_w or img_h < crop_y + crop_h:
            size_error += 1
            print(f"\n⚠️  {jpg_file.name}: size {img_w}x{img_h} too small")
            continue
        
        # 读取对应的png mask
        mask_file = data_path / f"{stem}.png"
        if not mask_file.exists():
            no_mask += 1
            continue
        
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            no_mask += 1
            continue
        
        # 裁剪
        cropped_img = crop_fixed_region(image, crop_x, crop_y, crop_w, crop_h)
        cropped_mask = crop_fixed_region(mask, crop_x, crop_y, crop_w, crop_h)
        
        # 验证裁剪后的尺寸
        if cropped_img.shape[0] != crop_h or cropped_img.shape[1] != crop_w:
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
    print(f"   No mask found: {no_mask}")
    print(f"   Size error: {size_error}")
    print(f"\n💾 Output:")
    print(f"   Images: {save_path / 'images'}")
    print(f"   Masks:  {save_path / 'masks'}")
    print(f"   All images are {crop_w}x{crop_h}")
    print("="*60)

# ============ 运行 ============
if __name__ == '__main__':
    batch_crop_fixed(
        jpg_dir=r'D:\ddesktop\robotss\train-label-1467\train_xiangdong1164\datas',
        save_dir=r'D:\ddesktop\robotss\train-label-1467\train_xiangdong1164\im_mask',
        crop_x=0,       # 从左边界开始
        crop_y=60,      # 从y=60开始
        crop_w=720,     # 宽度720（整个宽度）
        crop_h=720      # 高度720（60到780）
    )