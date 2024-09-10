from PIL import Image
import cv2
import numpy as np

def readImage(image_path):
    pass

# input imageData 
# slide_wndow = (,)
# strides = (,)
# padding = (left, top, bottom, right)  # 0 1
class myImage():

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
        # {"(rowNum,colNum)": image_slide}
        self.imageSlide = {}

    def image_normalize(self, image):
        pass

    def image_resize(self, image, target_size):
        pass
    def image_fixedResize(self, image, padding):
        pass
    
    def reconstruct_image(self):
        pass
    

def update_slideDict(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)  # 执行函数的操作
        # get rows cols  ?



'''

class myImage():
    def __init__(self, image, target_size, slide_window, strides, padding):
        self.slideRowNum = 0
        self.slideColNum = 0
    def slideImage(self, image,, slide_window, strides, padding):
        self.slideRowNum = image slide row number
        self.slideColNum = image slide col number
        return struct contains  image_slide、image_slide-rowNum、image_slide-colNum
        # dict: self.image_slices[(row_num, col_num)] = slice

'''