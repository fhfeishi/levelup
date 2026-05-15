import cv2
from PIL import Image
import numpy as np

def read_colorimg(img_path, BGRout=True):
    pil_img = Image.open(img_path) 
    pil_color = pil_img.convert('RGB')
    cv_color = np.array(pil_color)
    if BGRout:
        cv_color = cv2.cvtColor(cv_color, cv2.COLOR_RGB2BGR)
    return cv_color

def read_gray(img_path):
    pil_img = Image.open(img_path)
    pil_gray = pil_img.convert('L')
    cv_gray = np.array(pil_gray)
    return cv_gray