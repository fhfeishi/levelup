import cv2
import PIL.Image as Image
import os
import numpy as np

def pil_to_cv2(pil_img):
    """Convert PIL Image to OpenCV format."""
    # Convert the PIL image to RGB format
    pil_img = pil_img.convert('RGB')
    # Convert the PIL image to a numpy array
    cv2_img = np.array(pil_img)
    # Convert RGB to BGR (OpenCV format)
    cv2_img = cv2_img[:, :, ::-1].copy()
    return cv2_img

def analyze_image(image_path):
    # Open image using PIL
    pil_img = Image.open(image_path)
    
    # Convert PIL image to OpenCV format
    cv2_img = pil_to_cv2(pil_img)
    
    # Analyze image information
    width, height = pil_img.size
    mode = pil_img.mode
    format = pil_img.format
    
    print(f"Image size (width x height): {width} x {height}")
    print(f"Image mode: {mode}")
    print(f"Image format: {format}")
    
    # Convert image to RGB format for unique color analysis
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a list of pixel values
    pixel_values = rgb_img.reshape((-1, 3))
    
    # Convert the list of pixel values to a set to find unique colors
    unique_colors = set(tuple(pixel) for pixel in pixel_values)
    unique_colors_count = len(unique_colors)
    print("unique color:", unique_colors)
    print(f"Number of unique colors: {unique_colors_count}")

if __name__ == '__main__':
    image_path = r'D:\Ddesktop\ppt\work\missing-2.5D-30NG-0518\IMG_9E710037_2024-04-10_14-29-25\IMG_9E710037_2024-04-10_14-29-25_Mask.png'
    analyze_image(image_path)

