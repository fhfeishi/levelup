import torch
import os
import random
import numpy as np
import cv2
import PIL.Image as Image
import torch.utils.data as data
from data_transforms import preprocess_input, cvtColor


"""
/proj/
    +---/dataset/
            +----/train/
            +----/test/
    +----/utils/ 
            +-----data_dataset.py  class: my_dataset
            +-----data_preprocess.py  data-preprocess,
            +-----utils.py      some-config, show-text-log-
            +-----utils_metrics   criterion
            +-----train_and_eval.py     criterion      evaluate     fit_one_epoch()
            +-----distributed_utils.py   # windows may no use
    +----train.py

"""

class MyDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None, num_classes=3):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_root = os.path.join(root, "dataset/train_data", "jpgs")
            self.mask_root = os.path.join(root, "dataset/train_data", "pngs")
        else:
            self.image_root = os.path.join(root, "dataset/test_data", "jpgs")
            self.mask_root = os.path.join(root, "dataset/test_data", "pngs")
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

        self.num_classes = num_classes

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        image = Image.open(image_path)
        assert image is not None, f"failed to read image: {image_path}"
        w, h = image.size

        target = Image.open(mask_path).convert('P')   # .convert('P') or not
        assert target is not None, f"failed to read mask: {mask_path}"

        # get a window_size_crop contains enough target pixels
        jpg_crop, png_crop = self.pillow_randomCrop(image, target, window_size=(640,640), min_targetPixelNum_th=1000)
        

        # img-transforms
        jpg_crop, png_crop = self.my_img_augment(jpg_crop, png_crop, window_size=(640,640), jitter=.3, hue=.1, sat=0.7, val=0.3, random=True)

        #  ToTensor
        jpg_crop = cvtColor(jpg_crop)
        jpg_crop = np.transpose(preprocess_input(np.array(jpg_crop, np.float64)), [2, 0, 1])
        png_crop = np.array(png_crop)  # png_crop: (ww, wh)
        png_crop[png_crop >= self.num_classes] = self.num_classes   # may no use

        # tensor-transforms
        # sth


        return jpg_crop, png_crop

    def __len__(self):
        return len(self.images_path)

    def pillow_randomCrop(self, image, mask, window_size=(640,640), min_targetPixelNum_th=4000):
        w, h = image.size 
        ww, wh = window_size
        if w > ww and h > wh:
            while True:
                x = random.randint(0, w-ww)
                y = random.randint(0, h-wh)

                # cv2-image: h, w, c
                jpg_crop = image.crop((x, y, x+ww, y+wh))
                png_crop = mask.crop((x, y, x+ww, y+wh))

                png_crop_array = np.array(png_crop)

                non_background_pixels = np.sum(png_crop_array != 0)
                if non_background_pixels > min_targetPixelNum_th:
                    break


    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets
    
def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    path = "../dataset"    # True
    print(os.path.exists(path))

    train_dataset = MyDataset("../", train=True)
    print(len(train_dataset))

    val_dataset = MyDataset("../", train=False)
    print(len(val_dataset))

    i, t, l = train_dataset[0]