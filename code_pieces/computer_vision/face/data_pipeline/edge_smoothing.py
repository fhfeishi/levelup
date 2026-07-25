import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 边缘平滑


def smooth_small_spots(mask, min_area=3, smooth_strength='light'):
    """
    温和地平滑斑点，保留所有可能的真实斑点（包括只有4像素的小点）
    
    Args:
        mask: 输入mask图像
        min_area: 最小保留面积（像素），默认3（保留几乎所有点）
        smooth_strength: 平滑强度 'light', 'medium', 'strong'
    """
    result_mask = mask.copy()
    
    # 1. 温和的闭运算：轻微合并相邻的白点（核很小，避免过度合并）
    if smooth_strength in ['medium', 'strong']:
        kernel_close = np.ones((3, 3), np.uint8)
        result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # 2. 连通区域分析：过滤极小的噪点，但保留4像素以上的点
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result_mask, connectivity=8)
    
    # 创建清理后的mask
    cleaned_mask = np.zeros_like(result_mask)
    kept_regions = 0
    
    for i in range(1, num_labels):  # 跳过背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:  # 保留满足最小面积的所有区域
            cleaned_mask[labels == i] = 255
            kept_regions += 1
    
    # 3. 边缘平滑（根据强度选择不同方法）
    if smooth_strength == 'light':
        # 轻度平滑：只用小核高斯模糊
        blurred = cv2.GaussianBlur(cleaned_mask, (3, 3), 0.5)
        final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
        
    elif smooth_strength == 'medium':
        # 中度平滑：中等核高斯模糊
        blurred = cv2.GaussianBlur(cleaned_mask, (5, 5), 1.0)
        final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
        
    else:  # 'strong'
        # 强度平滑：大核高斯模糊 + 轻微形态学平滑
        blurred = cv2.GaussianBlur(cleaned_mask, (5, 5), 1.5)
        final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
        
        # 额外的形态学平滑（可选）
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_smooth, iterations=1)
    
    return final_mask, kept_regions

def analyze_mask_spots(mask, min_area=3):
    """分析mask中的斑点信息"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    spots_info = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y = centroids[i]
            spots_info.append({
                'id': i,
                'area': area,
                'center': (int(x), int(y)),
                'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            })
    
    return spots_info

def process_single_mask(image_path, min_area=3, smooth_strength='light', visualize=False):
    """处理单个mask"""
    mask = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None
    
    # 统计原始斑点
    original_spots = analyze_mask_spots(mask, min_area)
    
    # 平滑处理
    result_mask, kept_regions = smooth_small_spots(mask, min_area, smooth_strength)
    
    # 统计处理后的斑点
    result_spots = analyze_mask_spots(result_mask, min_area)
    
    stats = {
        'original_count': len(original_spots),
        'result_count': len(result_spots),
        'original_areas': [s['area'] for s in original_spots],
        'result_areas': [s['area'] for s in result_spots]
    }
    
    if visualize:
        # 创建可视化对比图
        vis_original = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis_result = cv2.cvtColor(result_mask, cv2.COLOR_GRAY2BGR)
        
        # 在原始图上标注斑点
        for spot in original_spots:
            x, y = spot['center']
            cv2.circle(vis_original, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(vis_original, f"{spot['area']}", (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # 在结果图上标注斑点
        for spot in result_spots:
            x, y = spot['center']
            cv2.circle(vis_result, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(vis_result, f"{spot['area']}", (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # 添加统计信息
        cv2.putText(vis_original, f"Spots: {len(original_spots)}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_result, f"Spots: {len(result_spots)}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        comparison = np.hstack([vis_original, vis_result])
        return result_mask, stats, comparison
    
    return result_mask, stats

def batch_process_masks(input_dir, output_dir=None, min_area=3, 
                       smooth_strength='light', preview_first=True):
    """
    批量处理mask - 鲁棒版本
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（默认原地保存）
        min_area: 最小保留面积（像素），3可以保留4像素的点
        smooth_strength: 平滑强度 'light'(推荐), 'medium', 'strong'
        preview_first: 是否先预览第一张
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有png图片
    image_files = sorted(list(input_path.glob('*.png')))
    
    if not image_files:
        print(f"❌ No PNG files found in {input_dir}")
        return
    
    print(f"{'='*70}")
    print(f"Found {len(image_files)} images")
    print(f"Settings:")
    print(f"  - Min spot area: {min_area} pixels (保留{min_area}像素及以上的点)")
    print(f"  - Smooth strength: {smooth_strength}")
    print(f"{'='*70}\n")
    
    # 预览第一张
    if preview_first and len(image_files) > 0:
        print("🔍 Generating preview for first image...")
        result, stats, comparison = process_single_mask(
            image_files[0], min_area, smooth_strength, visualize=True
        )
        
        if result is not None:
            preview_path = output_path / 'preview_comparison.png'
            cv2.imwrite(str(preview_path), comparison)
            
            print(f"\n📊 Preview statistics:")
            print(f"   Original spots: {stats['original_count']}")
            print(f"   After smoothing: {stats['result_count']}")
            print(f"   Original areas: {stats['original_areas']}")
            print(f"   Result areas: {stats['result_areas']}")
            print(f"\n💾 Preview saved to: {preview_path}")
            print(f"\n{'='*70}")
            print("Continue with batch processing? (y/n): ", end='')
            
            response = input().strip().lower()
            if response != 'y':
                print("❌ Cancelled.")
                return
            print()
    
    # 批量处理
    success_count = 0
    all_stats = []
    warnings = []
    
    for img_path in tqdm(image_files, desc="Processing"):
        result_mask, stats = process_single_mask(
            img_path, min_area, smooth_strength, visualize=False
        )
        
        if result_mask is None:
            warnings.append(f"Failed to process: {img_path.name}")
            continue
        
        all_stats.append(stats)
        
        # 检查是否丢失了斑点
        if stats['result_count'] < stats['original_count']:
            diff = stats['original_count'] - stats['result_count']
            warnings.append(f"{img_path.name}: Lost {diff} spot(s) "
                          f"({stats['original_count']} → {stats['result_count']})")
        
        # 生成新文件名：移除 '_mask'
        new_name = img_path.stem.replace('_mask', '')
        
        # 保存
        output_file = output_path / f'{new_name}.png'
        cv2.imwrite(str(output_file), result_mask)
        success_count += 1
    
    # 统计报告
    print(f"\n{'='*70}")
    print(f"✅ Successfully processed {success_count}/{len(image_files)} images")
    
    if all_stats:
        original_counts = [s['original_count'] for s in all_stats]
        result_counts = [s['result_count'] for s in all_stats]
        
        print(f"\n📊 Overall statistics:")
        print(f"   Spot counts before smoothing:")
        print(f"      Min: {min(original_counts)}, Max: {max(original_counts)}, "
              f"Avg: {np.mean(original_counts):.1f}")
        print(f"   Spot counts after smoothing:")
        print(f"      Min: {min(result_counts)}, Max: {max(result_counts)}, "
              f"Avg: {np.mean(result_counts):.1f}")
        
        # 统计各种斑点数量的图像
        from collections import Counter
        count_distribution = Counter(result_counts)
        print(f"\n   Distribution:")
        for count in sorted(count_distribution.keys()):
            print(f"      {count} spots: {count_distribution[count]} images")
    
    # 警告信息
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for w in warnings[:10]:  # 只显示前10个
            print(f"   {w}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings)-10} more warnings")
    
    print(f"\n💾 Output directory: {output_path}")
    print(f"{'='*70}")

# ============ 使用示例 ============
if __name__ == '__main__':
    # 配置
    INPUT_DIR = r'D:\ddesktop\robotss\train-label-1467\train_xiangdong1164'
    OUTPUT_DIR = None  # None表示原地保存
    
    # 参数设置
    MIN_AREA = 2          # 保留3像素及以上的点（可以捕获4像素的点）
    SMOOTH_STRENGTH = 'light'  # 'light' | 'medium' | 'strong'
    
    # 执行
    batch_process_masks(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        min_area=MIN_AREA,
        smooth_strength=SMOOTH_STRENGTH,
        preview_first=True  # 建议先预览
    )


