
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 

# mediapipe    ## InsightFace 慢一点     ## Face-Parsing 精确分割、更慢一些

class FaceDatasetCropper:
    def __init__(self, margin=0.3):
        """
        初始化人脸检测器（适配MediaPipe 0.10+）
        
        Args:
            margin: 扩展边距比例（0.3表示扩展30%）
        """
        self.margin = margin
        
        # 下载模型文件（首次运行会自动下载）
        # 如果下载失败，手动下载：https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
        model_path = 'blaze_face_short_range.tflite'
        
        # 检查模型是否存在，如果不存在则下载
        if not Path(model_path).exists():
            print("Downloading face detection model...")
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
            urllib.request.urlretrieve(url, model_path)
            print(f"✓ Model downloaded: {model_path}")
        
        # 创建检测器选项
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5
        )
        
        # 创建检测器
        self.detector = vision.FaceDetector.create_from_options(options)
    
    def detect_face_bbox(self, image):
        """
        检测人脸并返回bbox
        
        Returns:
            (x, y, w, h) 或 None
        """
        h, w = image.shape[:2]
        
        # 转换为MediaPipe Image格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 检测人脸
        ss = time.perf_counter()
        detection_result = self.detector.detect(mp_image)
        
        
        if not detection_result.detections:
            return None
        
        # 获取第一个人脸的bbox
        detection = detection_result.detections[0]
        bbox = detection.bounding_box
        
        # 获取坐标
        x = bbox.origin_x
        y = bbox.origin_y
        fw = bbox.width
        fh = bbox.height
        
        # 扩展边距
        margin_w = int(fw * self.margin)
        margin_h = int(fh * self.margin)
        
        x = max(0, x - margin_w)
        y = max(0, y - margin_h)
        fw = min(w - x, fw + 2 * margin_w)
        fh = min(h - y, fh + 2 * margin_h)
        
        return (x, y, fw, fh)
    
    def crop_image(self, image, bbox):
        """根据bbox裁剪图像"""
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    
    def process_dataset(self, image_dir, mask_dir, save_dir, 
                       bbox_txt='face_bbox.txt', preview_first=3):
        """
        批量处理数据集：检测人脸、裁剪图片和mask
        
        Args:
            image_dir: 原始图片目录
            mask_dir: 原始mask目录
            save_dir: 输出目录
            bbox_txt: bbox文件名
            preview_first: 预览前N张
        """
        image_path = Path(image_dir)
        mask_path = Path(mask_dir)
        save_path = Path(save_dir)
        
        # 创建输出目录
        (save_path / 'images').mkdir(parents=True, exist_ok=True)
        (save_path / 'masks').mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
            image_files.extend(image_path.glob(ext))
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return
        
        print(f"{'='*70}")
        print(f"Found {len(image_files)} images")
        print(f"Margin: {self.margin * 100}%")
        print(f"{'='*70}\n")
        
        # 第一步：检测所有人脸bbox
        print("🔍 Step 1: Detecting faces...")
        bboxes = {}
        no_face = []
        ss = time.perf_counter()
        for img_file in tqdm(image_files, desc="Detecting"):
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            try:
                bbox = self.detect_face_bbox(image)
                
                if bbox:
                    bboxes[img_file.stem] = bbox
                else:
                    no_face.append(img_file.stem)
            except Exception as e:
                print(f"\n⚠️  Error processing {img_file.name}: {e}")
                no_face.append(img_file.stem)
        
        print(f"\n✓ Detected {len(bboxes)} faces")
        print(f"✗ No face detected: {len(no_face)}")
        
        if not bboxes:
            print("❌ No faces detected in any image!")
            return
        
        tt = time.perf_counter()
        print("time cost", f"{tt-ss :.5f} s")
        
        # 保存bbox到txt
        bbox_file = save_path / bbox_txt
        with open(bbox_file, 'w') as f:
            for stem, (x, y, w, h) in sorted(bboxes.items()):
                f.write(f"{stem} {x},{y},{w},{h}\n")
        
        print(f"✓ BBox saved to: {bbox_file}")
        
        # 预览模式
        if preview_first > 0:
            print(f"\n🔍 Step 2: Preview mode (first {preview_first} images)...")
            preview_dir = save_path / 'preview'
            preview_dir.mkdir(exist_ok=True)
            
            preview_count = 0
            for stem, bbox in list(bboxes.items())[:preview_first]:
                # 查找图片文件
                img_file = None
                for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                    potential = image_path / f"{stem}{ext}"
                    if potential.exists():
                        img_file = potential
                        break
                
                if not img_file:
                    continue
                
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                # 查找mask
                mask_file = None
                for ext in ['.png', '.PNG']:
                    potential = mask_path / f"{stem}{ext}"
                    if potential.exists():
                        mask_file = potential
                        break
                
                mask = None
                if mask_file:
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                
                # 裁剪
                x, y, w, h = bbox
                cropped_img = self.crop_image(image, bbox)
                
                # 可视化：原图+bbox框+裁剪结果
                vis_original = image.copy()
                cv2.rectangle(vis_original, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(vis_original, f"{w}x{h}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Resize用于显示
                scale = 400 / max(vis_original.shape[:2])
                vis_original = cv2.resize(vis_original, 
                                         (int(vis_original.shape[1]*scale), 
                                          int(vis_original.shape[0]*scale)))
                
                scale_crop = 400 / max(cropped_img.shape[:2])
                vis_cropped = cv2.resize(cropped_img,
                                        (int(cropped_img.shape[1]*scale_crop),
                                         int(cropped_img.shape[0]*scale_crop)))
                
                # 拼接对比图
                h_max = max(vis_original.shape[0], vis_cropped.shape[0])
                vis_original = cv2.copyMakeBorder(vis_original, 0, h_max-vis_original.shape[0],
                                                  0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
                vis_cropped = cv2.copyMakeBorder(vis_cropped, 0, h_max-vis_cropped.shape[0],
                                                 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
                
                comparison = np.hstack([vis_original, vis_cropped])
                cv2.imwrite(str(preview_dir / f'preview_{preview_count+1}.jpg'), comparison)
                
                print(f"✓ {stem}: bbox=({x},{y},{w},{h}), size={w}x{h}")
                preview_count += 1
            
            print(f"\nPreview saved to: {preview_dir}")
            print(f"{'='*70}")
            print("Continue with full processing? (y/n): ", end='')
            
            response = input().strip().lower()
            if response != 'y':
                print("❌ Cancelled.")
                return
            print()
        
        # 第三步：批量裁剪
        print("✂️  Step 3: Cropping images and masks...")
        success = 0
        no_mask = 0
        
        for stem, bbox in tqdm(bboxes.items(), desc="Cropping"):
            # 查找图片文件
            img_file = None
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                potential = image_path / f"{stem}{ext}"
                if potential.exists():
                    img_file = potential
                    break
            
            if not img_file:
                continue
            
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # 查找mask
            mask_file = None
            for ext in ['.png', '.PNG']:
                potential = mask_path / f"{stem}{ext}"
                if potential.exists():
                    mask_file = potential
                    break
            
            mask = None
            if mask_file:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            else:
                no_mask += 1
            
            # 裁剪图片
            cropped_img = self.crop_image(image, bbox)
            
            # 裁剪mask
            if mask is not None:
                cropped_mask = self.crop_image(mask, bbox)
                cv2.imwrite(str(save_path / 'masks' / f"{stem}.png"), cropped_mask)
            
            # 保存图片
            cv2.imwrite(str(save_path / 'images' / img_file.name), cropped_img)
            
            success += 1
        
        # 统计报告
        print(f"\n{'='*70}")
        print(f"✅ Processing completed!")
        print(f"   Successfully cropped: {success}/{len(image_files)}")
        print(f"   No face detected: {len(no_face)}")
        print(f"   No mask found: {no_mask}")
        
        if bboxes:
            widths = [w for _, _, w, h in bboxes.values()]
            heights = [h for _, _, w, h in bboxes.values()]
            print(f"\n📊 Cropped size statistics:")
            print(f"   Width:  min={min(widths)}, max={max(widths)}, avg={int(np.mean(widths))}")
            print(f"   Height: min={min(heights)}, max={max(heights)}, avg={int(np.mean(heights))}")
        
        if no_face:
            print(f"\n⚠️  Images without face ({len(no_face)}):")
            for stem in no_face[:10]:
                print(f"   {stem}")
            if len(no_face) > 10:
                print(f"   ... and {len(no_face)-10} more")
        
        print(f"\n💾 Output:")
        print(f"   BBox file: {bbox_file}")
        print(f"   Images: {save_path / 'images'}")
        print(f"   Masks: {save_path / 'masks'}")
        print(f"{'='*70}")
        
# ============ 使用示例 ============
if __name__ == '__main__':
    # 创建裁剪器
    cropper = FaceDatasetCropper(margin=0.2)  # 扩展30%边距
    
    # 处理数据集
    cropper.process_dataset(
        image_dir=r'D:\ddesktop\robotss\train-label-1467\train-data-robot303\train_data',
        mask_dir=r'D:\ddesktop\robotss\train-label-1467\train-data-robot303\train_cleaned',
        save_dir=r'D:\ddesktop\robotss\train-label-1467\train_data\robot_data',
        bbox_txt=r'D:\ddesktop\robotss\codespace\robot_face_bbox.txt',
        preview_first=0  # 先预览前3张
    )

