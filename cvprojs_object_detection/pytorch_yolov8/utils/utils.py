import numpy as np
from PIL import Image
import cv2


def pil_to_cv2(pil_img):
    cv2_img = np.array(pil_img)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    return cv2_img
def cv2_to_pil(cv2_img):
    pil_image = Image.fromarray(cv2_img)
    return pil_image


# 将图像转换成'RGB'图像
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def resize_pil_img(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw    = int(iw*scale)
        nh    = int(ih*scale)

        image = image.resize((nw,nh), Image.Resampling.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)/2, (h-nh)/2))
    else:
        new_image = image.resize((w,h), Image.Resampling.BICUBIC)
    return new_image

# normalize image data
def preprocess_input(image):
    image /= 255.0
    return image


# get class-names len(classes-names)
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)










