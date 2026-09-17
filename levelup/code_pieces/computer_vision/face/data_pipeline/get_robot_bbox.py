# 得到机器人头的bbox   xywh
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def detect_robot_head_bbox(image, method='auto'):
    """
    检测机器人头部的bbox
    
    Args:
        image: 输入图像
        method: 检测方法 'threshold', 'edge', 'grabcut', 'auto'
    
    Returns:
        (x, y, w, h) 或 None
    """
    h, w = image.shape[:2]
    
    if method == 'threshold' or method == 'auto':
        # 方法1：基于阈值的检测（推荐用于简单背景）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Otsu自动阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 如果前景是黑色，反转
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # 形态学操作清理
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
    elif method == 'edge':
        # 方法2：基于边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=3)
        
        # 填充轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        binary = np.zeros_like(gray)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(binary, [largest], -1, 255, -1)
    
    elif method == 'grabcut':
        # 方法3：GrabCut（更精确但慢）
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
        
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            binary = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        except:
            return None
    
    # 查找最大的连通区域
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 过滤太小的区域（小于图像面积的1%）
    min_area = (w * h) * 0.01
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        return None
    
    # 获取最大轮廓的bbox
    largest_contour = max(valid_contours, key=cv2.contourArea)
    x, y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
    
    return (x, y, bbox_w, bbox_h)

def visualize_bbox(image, bbox, title="Detection Result"):
    """可视化bbox"""
    vis = image.copy()
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis, f"Size: {w}x{h}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis

def batch_detect_and_save(image_dir, output_txt='bbox.txt', method='auto', 
                          preview_first=True, save_visualizations=False):
    """
    批量检测机器人头部bbox并保存
    
    Args:
        image_dir: 图片目录
        output_txt: 输出文本文件路径
        method: 检测方法 'threshold'(推荐), 'edge', 'grabcut', 'auto'
        preview_first: 是否先预览前几张
        save_visualizations: 是否保存可视化结果
    """
    image_path = Path(image_dir)
    
    # 支持的图片格式
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(image_path.glob(ext))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"{'='*70}")
    print(f"Found {len(image_files)} images")
    print(f"Detection method: {method}")
    print(f"{'='*70}\n")
    
    # 预览模式
    if preview_first:
        print("🔍 Preview mode: Processing first 3 images...\n")
        preview_dir = image_path / 'preview_bbox'
        preview_dir.mkdir(exist_ok=True)
        
        for i, img_file in enumerate(image_files[:3]):
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            bbox = detect_robot_head_bbox(image, method)
            
            if bbox:
                x, y, w, h = bbox
                print(f"✓ {img_file.name}")
                print(f"  BBox: x={x}, y={y}, w={w}, h={h}")
                print(f"  Size: {w}x{h}\n")
                
                # 保存可视化
                vis = visualize_bbox(image, bbox)
                cv2.imwrite(str(preview_dir / f'preview_{i+1}.jpg'), vis)
            else:
                print(f"❌ {img_file.name}: Detection failed\n")
        
        print(f"Preview images saved to: {preview_dir}")
        print(f"{'='*70}")
        print("Continue with full batch processing? (y/n): ", end='')
        
        response = input().strip().lower()
        if response != 'y':
            print("❌ Cancelled.")
            return
        print()
    
    # 批量处理
    results = []
    failed = []
    sizes = []  # 记录宽高统计
    
    # 创建可视化目录（如果需要）
    if save_visualizations:
        vis_dir = image_path / 'bbox_visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    for img_file in tqdm(image_files, desc="Processing"):
        image = cv2.imread(str(img_file))
        if image is None:
            failed.append((img_file.stem, "Failed to read image"))
            continue
        
        bbox = detect_robot_head_bbox(image, method)
        
        if bbox:
            x, y, w, h = bbox
            results.append((img_file.stem, x, y, w, h))
            sizes.append((w, h))
            
            # 保存可视化
            if save_visualizations:
                vis = visualize_bbox(image, bbox)
                cv2.imwrite(str(vis_dir / f'{img_file.stem}_bbox.jpg'), vis)
        else:
            failed.append((img_file.stem, "No bbox detected"))
    
    # 保存到txt文件
    output_path = Path(output_txt)
    with open(output_path, 'w') as f:
        for stem, x, y, w, h in results:
            f.write(f"{stem} {x},{y},{w},{h}\n")
    
    # 统计报告
    print(f"\n{'='*70}")
    print(f"✅ Processing completed!")
    print(f"   Successfully detected: {len(results)}/{len(image_files)}")
    print(f"   Failed: {len(failed)}")
    
    if sizes:
        widths = [w for w, h in sizes]
        heights = [h for w, h in sizes]
        
        print(f"\n📊 Size statistics:")
        print(f"   Width:  min={min(widths)}, max={max(widths)}, avg={int(np.mean(widths))}")
        print(f"   Height: min={min(heights)}, max={max(heights)}, avg={int(np.mean(heights))}")
        print(f"   Common sizes:")
        
        # 统计最常见的尺寸
        from collections import Counter
        size_counts = Counter(sizes)
        for (w, h), count in size_counts.most_common(5):
            print(f"      {w}x{h}: {count} images")
    
    if failed:
        print(f"\n⚠️  Failed images ({len(failed)}):")
        for stem, reason in failed[:10]:
            print(f"   {stem}: {reason}")
        if len(failed) > 10:
            print(f"   ... and {len(failed)-10} more")
    
    print(f"\n💾 BBox file saved to: {output_path}")
    if save_visualizations:
        print(f"📷 Visualizations saved to: {vis_dir}")
    print(f"{'='*70}")
    
    return results

def load_bbox_from_txt(bbox_txt):
    """从txt文件加载bbox信息"""
    bboxes = {}
    with open(bbox_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                stem = parts[0]
                coords = parts[1].split(',')
                if len(coords) == 4:
                    x, y, w, h = map(int, coords)
                    bboxes[stem] = (x, y, w, h)
    return bboxes

def crop_images_by_bbox(image_dir, bbox_txt, output_dir, padding=0):
    """
    根据bbox.txt裁剪图片
    
    Args:
        image_dir: 原始图片目录
        bbox_txt: bbox文件路径
        output_dir: 输出目录
        padding: 裁剪时的边距（像素）
    """
    image_path = Path(image_dir)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载bbox
    bboxes = load_bbox_from_txt(bbox_txt)
    print(f"Loaded {len(bboxes)} bboxes from {bbox_txt}")
    
    # 查找图片
    image_files = []
    # for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    for ext in ['*.jpg']:
        image_files.extend(image_path.glob(ext))
    
    success = 0
    for img_file in tqdm(image_files, desc="Cropping"):
        if img_file.stem not in bboxes:
            continue
        
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        x, y, w, h = bboxes[img_file.stem]
        
        # 添加padding并确保不越界
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # 裁剪
        cropped = image[y1:y2, x1:x2]
        
        # 保存
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), cropped)
        success += 1
    
    print(f"✓ Cropped {success} images to {output_path}")

# ============ 使用示例 ============
if __name__ == '__main__':
    # ========== 步骤1：检测bbox并保存 ==========
    IMAGE_DIR = r'D:\ddesktop\robotss\train-label-1467\train-data-robot303\train_data'
    BBOX_TXT = 'bbox.txt'
    
    # 方法选择：
    # 'threshold' - 基于阈值（推荐，适合简单背景）
    # 'edge' - 基于边缘检测
    # 'grabcut' - GrabCut算法（精确但慢）
    # 'auto' - 自动选择（默认threshold）
    
    batch_detect_and_save(
        image_dir=IMAGE_DIR,
        output_txt=BBOX_TXT,
        method='threshold',      # 推荐
        preview_first=True,      # 先预览前3张
        save_visualizations=False # 是否保存可视化结果
    )
    
    # ========== 步骤2（可选）：根据bbox裁剪图片 ==========
    # crop_images_by_bbox(
    #     image_dir=IMAGE_DIR,
    #     bbox_txt=BBOX_TXT,
    #     output_dir='path/to/cropped_images',
    #     padding=20  # 裁剪时额外保留20像素边距
    # )



