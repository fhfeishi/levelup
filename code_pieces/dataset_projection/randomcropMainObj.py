import random
import numpy as np


def od_randomCropA(image, bboxes, crop_size=(640,640), min_objectInArea_ratio=0.6):
    
    try:
        h, w, _ = image.shape
    except:
        w, h = image.size
        
    cw, ch = crop_size
    assert h >= ch and w >= cw, f"Image size must be at least {cw}x{ch}"

    while True:
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)
        x2 = x1 + cw
        y2 = y1 + ch

        crop_bbox = [x1, y1, x2, y2]

        # 记录相对于crop的bbox
        new_bboxes = []
        for bbox in bboxes:
            bbox_x1 = max(crop_bbox[0], bbox[0])
            bbox_y1 = max(crop_bbox[1], bbox[1])
            bbox_x2 = min(crop_bbox[2], bbox[2])
            bbox_y2 = min(crop_bbox[3], bbox[3])

            intersection_area = max(0, bbox_x2 - bbox_x1) * max(0, bbox_y2 - bbox_y1)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # 如果 bbox_area * min_objectInArea_ratio > crop_size
            # 目标检测数据集 xml-bbox  area-size： count：  
            # 对于超大目标+小目标的，代码还需要额外处理
            if bbox_area * min_objectInArea_ratio > cw * ch:
                # 这里就减小 ratio 裁剪 超大目标的部分区域， 
                # 网络训练的好的话，能否通过这些部分区域学习到超大目标的全部区域？
                new_ratio = (cw*ch)/bbox_area - 0.1
                if intersection_area / bbox_area > new_ratio:
                    new_bboxes.append([bbox_x1 - x1, bbox_y1 - y1, bbox_x2 - x1, bbox_y2 - y1])
            # bbox_area * min_objectInArea_ratio =< crop_size
            else:
                if intersection_area / bbox_area > min_objectInArea_ratio:
                    new_bboxes.append([bbox_x1 - x1, bbox_y1 - y1, bbox_x2 - x1, bbox_y2 - y1])

        if len(new_bboxes) > 0:
            cropped_image = image[y1:y2, x1:x2]
            return cropped_image, new_bboxes



def od_randomCropB(image, bboxes, crop_size=(640, 640), min_object_cover=0.7, max_attempts=100):
        """
        随机裁剪图像，并确保主要目标至少被覆盖一定比例。
        """
        try:
            width, height = image.size
        except:
            height, width, _ = image.shape
            
        crop_width, crop_height = crop_size

        # 确保裁剪尺寸不会超过原始图像尺寸
        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)

        for _ in range(max_attempts):
            # 随机选择裁剪的位置
            x = np.random.randint(0, width - crop_width + 1)
            y = np.random.randint(0, height - crop_height + 1)
            crop_region = np.array([x, y, x + crop_width, y + crop_height])

            # 计算交叉区域
            new_bboxes = []
            for bbox in bboxes:
                overlap_x1 = max(crop_region[0], bbox[0])
                overlap_y1 = max(crop_region[1], bbox[1])
                overlap_x2 = min(crop_region[2], bbox[2])
                overlap_y2 = min(crop_region[3], bbox[3])

                overlap_width = max(0, overlap_x2 - overlap_x1)
                overlap_height = max(0, overlap_y2 - overlap_y1)
                overlap_area = overlap_width * overlap_height

                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                # 检查重叠面积是否满足要求
                if (overlap_area / bbox_area) >= min_object_cover:
                    new_bbox = [overlap_x1 - x, overlap_y1 - y, overlap_x2 - x, overlap_y2 - y, bbox[4]]
                    new_bboxes.append(new_bbox)

            if new_bboxes:
                # 如果找到了满足条件的裁剪，进行裁剪
                cropped_image = image.crop((x, y, x + crop_width, y + crop_height))
                return cropped_image, np.array(new_bboxes)

        # 如果没有找到合适的裁剪，返回原图
        return image, bboxes


