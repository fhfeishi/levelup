import time
import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
import io
import matplotlib.pyplot as plt
import random
import xml.etree.cElementTree as ET
from xml.dom.minidom import Document
from tqdm import tqdm

class imageSlide():
    def __init__(self):
        # 输入图片的高度宽度
        self.width = None
        self.height = None
        # 缩放之后，图片的高度、宽度
        self.scaled_height = None
        self.scaled_width = None
        # 滑窗大小
        self.crop_size = None
        # 滑窗移动步长：宽度方向 高度方向
        self.stride_width = None
        self.stride_height = None
        # 对原图下方补零长度 右方补零长度
        self.padding_bottom = 0
        self.padding_right = 0
        self.padding_top = 0
        self.padding_left = 0
        # 滑窗的移动步数宽度/高度方向 + 1、滑窗截图的行/列数 + 1
        self.cols_num = None
        self.rows_num = None
        # imageblock row_index col_index
        self._ImageBlock = {}
        
    def paddedImage_crop(self, image, crop_size=640, stride=(0,0)):
        width, height = image.size
        # crop size
        self.crop_size = crop_size
        # stride_width  stride_height
        self.stride_width, self.stride_height = stride
        
        # rows num
        height_length = height - crop_size
        if height_length % self.stride_height != 0:
            num_rows = int(height_length // self.stride_height) + 1
            self.rows_num = num_rows + 1
        else:
            num_rows = int(height_length / self.stride_height)
            self.rows_num = num_rows + 1
            
        # cols num
        width_length = width - crop_size
        if width_length % self.stride_width != 0:
            num_cols = int(width_length // self.stride_width) + 1
            self.cols_num = num_cols + 1
        else:
            num_cols = int(width_length // self.stride_width)
            self.cols_num = num_cols + 1

        # padding height, padding width
        padding_height = num_rows * self.stride_height + crop_size - height
        padding_width = num_cols * self.stride_width + crop_size - width
        self.padding_height = int(padding_height) + (int(padding_height) % 2)
        self.padding_width = int(padding_width) + (int(padding_width) % 2)
        
        # 右边和下面补零
        padded_image = ImageOps.expand(image, border=(
                0, 0, int(padding_width), int(padding_height)), fill=(0, 0, 0))
        
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                    start_row = r * self.stride_height
                    end_row = start_row + crop_size
                    start_col = c * self.stride_width
                    end_col = start_col + crop_size
                    block = padded_image.crop((start_col, start_row, end_col, end_row))
                    if block.size[0] == crop_size and block.size[1] == crop_size:  
                        self._ImageBlock[(r, c)] = block
        return self._ImageBlock
      
    

def randomCropBbox(opencvImage, bboxes, crop_size=(640,640), min_objectInArea_ratio=0.6, max_aatempts=50):
    h, w, _ = opencvImage.shape
    cw, ch = crop_size
    assert h >= ch and w >= cw, f"Image size must be at least {cw}x{ch}"

    for _ in range(max_aatempts):
        x1 = random.randint(0, (w-cw+1))
        y1 = random.randint(0, (h-ch+1))
        x2 = x1 + cw
        y2 = y1 + ch

        new_bboxes = []
        for bbox in bboxes:
            bbox_x1 = max(x1, bbox[0])
            bbox_y1 = max(y1, bbox[1])
            bbox_x2 = min(x2, bbox[2])
            bbox_y2 = min(y2, bbox[3])

            intersection_area = max(0, bbox_x2-bbox_x1) * max(0, bbox_y2-bbox_y1)
            bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])

            if intersection_area >= min_objectInArea_ratio * bbox_area:
                new_bboxes.append([bbox_x1 - x1, bbox_y1 - y1, bbox_x2 - x1, bbox_y2 - y1])

            if len(new_bboxes) > 0:
                cropped_image = opencvImage[y1:y2, x1:x2]
                return cropped_image, new_bboxes

    return opencvImage, bboxes

