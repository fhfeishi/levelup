import os
import PIL.Image
from PIL import Image
import numpy as np
import cv2
import random
import torch
import torch.utils.data as data

# filter warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# my dataset
class MyDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_root = os.path.join(root, "../DUTS-TR", "DUTS-TR-Image")
            self.mask_root = os.path.join(root, "../DUTS-TR", "DUTS-TR-Mask")
        else:
            self.image_root = os.path.join(root, "../DUTS-TE", "DUTS-TE-Image")
            self.mask_root = os.path.join(root, "../DUTS-TE", "DUTS-TE-Mask")
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check images and mask
        re_mask_names = []
        for p in image_names:
            mask_name = p.replace(".jpg", ".png")
            assert mask_name in mask_names, f"{p} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]

        # image_crop, mask_crop = self.opencv_random_crop(image_path, mask_path, window_size=(800, 640))
        image_crop, mask_crop = self.pillow_random_crop(image_path, mask_path, window_size=(320, 320))

        crop_label = self.rgb_to_class_index(mask_crop, key_rgb_map={'1': (30, 200, 40), '2': (60, 80, 210)})

        if self.transforms is not None:
            image_crop, crop_label = self.transforms(image_crop, crop_label)

        # print(crop_label.shape)  #

        return image_crop, crop_label

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def opencv_random_crop(image_path, mask_path, window_size=(800, 640), target_pix_num_th=1500):
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        print("opencv: image type 1", type(image))
        assert image is not None, f"failed to read image: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        print("opencv: image type 2", type(image))
        h, w, _ = image.shape

        target = cv2.imread(mask_path, flags=cv2.IMREAD_COLOR)
        print("opencv: target type 1", type(image))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        print("opencv: target type 2", type(image))
        assert target is not None, f"failed to read mask: {mask_path}"

        crop_width, crop_height = window_size
        # 确保图像尺寸足够大
        if w > crop_width and h > crop_height:
            while True:  # 无限循环  直到遇到下一个brake or  return
                # 随机选择裁剪起点
                x = random.randint(0, w - crop_width)
                y = random.randint(0, h - crop_height)

                # opencv使用切片进行裁剪, 默认是不会处理通道数的， cv2-image h, w, c
                image_crop = image[y:y + crop_height, x:x + crop_width]
                print("opencv: image_crop type", type(image_crop))

                mask_crop = target[y:y + crop_height, x:x + crop_width]
                print("opencv: mask_crop type", type(mask_crop))

                # 计算非背景像素的数量（假设背景标签为0）
                non_background_pixels = np.sum(np.any(mask_crop != 0, axis=2))
                print("non_background_pixels type",type(non_background_pixels))
                # 检查非背景像素是否满足要求
                if non_background_pixels > target_pix_num_th:
                    return image_crop, mask_crop

    @staticmethod
    def pillow_random_crop(image_path, mask_path, window_size=(800, 640), target_pix_num_th=8000):
        image = Image.open(image_path)
        print("pillow: image type", type(image))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
        assert image is not None, f"failed to read image: {image_path}"

        mask = Image.open(mask_path)
        print("pillow: mask type", type(mask))  # <class 'PIL.PngImagePlugin.PngImageFile'>
        assert mask is not None, f"failed to read mask: {mask_path}"

        w, h = image.size
        # print(image.size)

        crop_width, crop_height = window_size
        if w >= crop_width and h >= crop_height:
            while True:
                # 随机选择裁剪起点
                x = random.randint(0, w - crop_width)
                y = random.randint(0, h - crop_height)

                # 裁剪图像
                image_crop = image.crop((x, y, x + crop_width, y + crop_height))
                print("pillow: image_crop type", type(image_crop))  # pillow: image_crop type <class 'PIL.Image.Image'>
                mask_crop = mask.crop((x, y, x + crop_width, y + crop_height))
                print("pillow: mask_crop type", type(mask_crop))  # pillow: mask_crop type <class 'PIL.Image.Image'>

                # 将PNG裁剪图像的像素值转换为 numpy 数组以方便处理
                mask_array = np.array(mask_crop)
                print("pillow mask_array type", type(mask_array))   # numpy.ndarray

                # print("mask crop", mask_crop)

                # 计算非背景像素的数量（假设背景标签为0）
                non_background_pixels = np.sum(mask_array != 0)    # numpy.ndarray

                # 检查非背景像素是否满足要求
                if non_background_pixels > target_pix_num_th:
                    return image_crop, mask_crop

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets

# 补充的一个函数
def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

# my dataloader
class MyDataloader(object):
    pass


if __name__ == '__main__':
    from transforms import ToTensor, transCompose

    transforms = transCompose([
        ToTensor()
    ])

    train_dataset = MyDataset("./", train=True, transforms=transforms)
    print(len(train_dataset))

    val_dataset = MyDataset("./", train=False, transforms=transforms)
    print(len(val_dataset))

    i, t = train_dataset[0]


    # dataloader test  --if



