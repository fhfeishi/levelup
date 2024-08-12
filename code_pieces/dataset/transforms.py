import random
from typing import List, Union
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np


class transCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize(object):
    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
        self.size = size  # [h, w]
        self.resize_mask = resize_mask

    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if self.resize_mask is True:
            target = F.resize(target, self.size)

        return image, target


class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def pad_if_smaller(self, img, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        min_size = min(img.shape[-2:])
        if min_size < self.size:
            ow, oh = img.size
            padh = self.size - oh if oh < self.size else 0
            padw = self.size - ow if ow < self.size else 0
            img = F.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    def __call__(self, image, target):
        image = self.pad_if_smaller(image)
        target = self.pad_if_smaller(target)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

# transform
class Rescale_target(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image):
        return image.resize(self.target_size, Image.BICUBIC)

class Rescale_random(object):
    def __init__(self, min_size, max_size):
        assert isinstance(min_size, (int, float))
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, jpg, png):

        assert jpg.size == png.size
        w, h = jpg.size

        # randomly select size to resize
        if min(self.max_size, h, w) > self.min_size:
            self.target_size = np.random.randint(
                self.min_size, min(self.max_size, h, w)
            )
        else:
            self.target_size = self.min_size

        # calculates new size by keeping aspect ratio same
        if h > w:
            target_h, target_w = self.target_size * h / w, self.target_size
        else:
            target_h, target_w = self.target_size, self.target_size * w / h
        target_image = jpg.resize((target_w, target_h), Image.BICUBIC)
        target_mask = png.resize((target_w, target_h), Image.BICUBIC)


class ToTensor(object):
    """ conver image_data to tensor
    PIL.Image        --pillow  shape: h,w
    numpy.ndarray    --cv2  shape: h,w,c
    """
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, jpg, png):
        return self.totensor(jpg), self.totensor(png)


class RandomCrop_custom(object):
    """crop randomly the image-mask in"""
    def __init__(self, crop_size):
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.out_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.out_size = crop_size
        self.randomcrop = transforms.RandomCrop(self.out_size)

    def __call__(self, jpg, png):
        cropped_imgs = self.randomcrop(torch.cat((jpg, png), dim=0))  # axis=0

        return cropped_imgs[:3, :, ], cropped_imgs[3:, :, ]


class Normalize_custom(object):
    def __init__(self, mean, std):
        assert isinstance(mean, (float, tuple))
        if isinstance(mean, float):
            self.mean = (mean, mean, mean)
        else:
            assert len(mean) == 3
            self.mean = mean

        if isinstance(std, float):
            self.std = (std, std, std)
        else:
            assert len(std) == 3
            self.std = std

        self.normalize = transforms.Normalize(self.mean, self.std)
    def __call__(self, jpg, png):
        return self.normalize(jpg), self.normalize(png)



# daiding
class RandomCropTargetss(object):
    """designed for semantic segmentation: crop enought target pixels
    input: image  png  maybe:  torch.tensor   Pil.Image  numpy.ndarray

    """

    def __init__(self, image, png, win_size, TargetPixelNumTh=0.2):
        assert isinstance(win_size, (int, tuple))
        if isinstance(win_size, int):
            self.win_w, self.win_h = win_size
        elif len(win_size) == 2:
            self.win_w, self.win_h = win_size

# 由于是巡检图片，摄像机处于运动状态，待测目标是静止的，有空间上的移动变换
# 理论上来说要补充一些 平移变换、旋转变换（旋转的角度小一点）

# 向任意方向的移动
class boomMove(object):
    def __init__(self, max_translation=10):
        self.max_translation = max_translation

    def __call__(self, image, target):
        dx = random.randint(-self.max_translation, self.max_translation)
        dy = random.randint(-self.max_translation, self.max_translation)
        image = F.affine(image, angle=0, translate=[dx, dy], scale=1, shear=[0])
        if target is not None:
            target = F.affine(target, angle=0, translate=[dx, dy], scale=1, shear=[0])
        return image, target

# 小角度的旋转变换
class boomRotate(object):
    def __init__(self, max_angle=30):
        self.max_angle = max_angle

    def __call__(self, image, target):
        angle = random.uniform(-self.max_angle, self.max_angle)
        image = F.rotate(image, angle)
        if target is not None:
            target = F.rotate(target, angle)
        return image, target


