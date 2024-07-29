import cv2
import os
import PIL.Image as Image
import numpy as np
# check mask-png

def check_png(png_path, mode="opencv"):
    if mode == "opencv":
        png = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        unique_values = np.unique(png)
        print(unique_values)
    elif mode == "pillow":
        png = Image.open(png_path).convert('L')
        png_array = np.array(png)
        unique_values = np.unique(png_array)
        print(unique_values)
        
if __name__ == '__main__':
    
    png_path = r"D:\chyCodespace\project\jueyuanziboom\ss_u2net\DUTS-TR\DUTS-TR-Mask\image(10).png"
    check_png(png_path, mode="opencv")



