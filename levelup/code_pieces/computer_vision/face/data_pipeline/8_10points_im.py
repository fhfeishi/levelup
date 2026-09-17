
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# 找到10点、8点的图

def count_spots_in_mask(mask_path, min_area=3):
    """统计mask中的斑点数量"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    spot_count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            spot_count += 1
    
    return spot_count

def find_and_copy_abnormal_spots(mask_dir, image_dir, output_dir, 
                                 target_counts=[8, 10], min_area=3):
    """
    查找斑点数不正常的图片并复制
    
    Args:
        mask_dir: mask文件所在目录
        image_dir: 原始jpg图片所在目录
        output_dir: 输出目录
        target_counts: 要查找的斑点数量列表（如[8, 10]）
        min_area: 最小斑点面积
    """
    mask_path = Path(mask_dir)
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录结构
    output_path.mkdir(parents=True, exist_ok=True)
    for count in target_counts:
        (output_path / f"{count}_spots").mkdir(exist_ok=True)
    
    # 获取所有mask文件
    mask_files = sorted(list(mask_path.glob('*.png')))
    
    if not mask_files:
        print(f"❌ No PNG files found in {mask_dir}")
        return
    
    print(f"{'='*70}")
    print(f"Scanning {len(mask_files)} mask files...")
    print(f"Looking for images with {target_counts} spots")
    print(f"{'='*70}\n")
    
    # 存储结果
    results = {count: [] for count in target_counts}
    not_found_images = []
    not_found_masks = []
    
    # 扫描所有mask
    for mask_file in tqdm(mask_files, desc="Scanning"):
        spot_count = count_spots_in_mask(mask_file, min_area)
        
        if spot_count is None:
            continue
        
        if spot_count in target_counts:
            results[spot_count].append(mask_file)
    
    # 复制文件
    print(f"\n{'='*70}")
    print("📋 Found files:")
    for count in target_counts:
        print(f"   {count} spots: {len(results[count])} images")
    print(f"{'='*70}\n")
    
    total_copied = 0
    
    for count in target_counts:
        if not results[count]:
            continue
        
        print(f"\n📁 Processing {count}-spot images...")
        count_output_dir = output_path / f"{count}_spots"
        
        for mask_file in tqdm(results[count], desc=f"Copying {count}-spot"):
            stem = mask_file.stem
            
            # 查找对应的jpg图片
            # 尝试多种可能的命名方式
            possible_jpg_names = [
                stem + '.jpg',
                stem + '.JPG',
                stem.replace('_mask', '') + '.jpg',
                stem.replace('mask_', '') + '.jpg',
            ]
            
            jpg_file = None
            for jpg_name in possible_jpg_names:
                potential_path = image_path / jpg_name
                if potential_path.exists():
                    jpg_file = potential_path
                    break
            
            # 复制文件
            try:
                # 复制mask
                shutil.copy2(mask_file, count_output_dir / mask_file.name)
                
                # 复制jpg（如果找到）
                if jpg_file:
                    shutil.copy2(jpg_file, count_output_dir / jpg_file.name)
                    total_copied += 2
                else:
                    not_found_images.append(stem)
                    total_copied += 1
                    
            except Exception as e:
                print(f"\n⚠️  Error copying {stem}: {e}")
    
    # 输出报告
    print(f"\n{'='*70}")
    print(f"✅ Copy completed!")
    print(f"   Total files copied: {total_copied}")
    
    for count in target_counts:
        print(f"\n📊 {count}-spot images:")
        print(f"   Count: {len(results[count])}")
        print(f"   Location: {output_path / f'{count}_spots'}")
    
    if not_found_images:
        print(f"\n⚠️  Warning: {len(not_found_images)} jpg images not found:")
        for stem in not_found_images[:10]:
            print(f"   {stem}.jpg")
        if len(not_found_images) > 10:
            print(f"   ... and {len(not_found_images)-10} more")
    
    print(f"\n💾 Output directory: {output_path}")
    print(f"{'='*70}")
    
    return results

def quick_find_abnormal_spots(mask_dir, image_dir, output_dir='path/to/ggims'):
    """
    快速查找8点和10点的图片
    """
    find_and_copy_abnormal_spots(
        mask_dir=mask_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        target_counts=[8, 10],
        min_area=3
    )

# ============ 使用方法1：快速版本 ============
if __name__ == '__main__':
    # 配置路径
    MASK_DIR = r'D:\ddesktop\robotss\train-label-1467\train_xiangdong1164'      # mask文件所在目录（处理后的）
    IMAGE_DIR = r'D:\ddesktop\robotss\train-label-1467\train_xiangdong1164'  # 原始jpg图片所在目录
    OUTPUT_DIR = r'D:\ddesktop\robotss\train-label-1467\8-10-points'  # 输出目录
    
    # 执行
    quick_find_abnormal_spots(
        mask_dir=MASK_DIR,
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR
    )
