import random
from typing import List, Union
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T


class Compose(object):
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


