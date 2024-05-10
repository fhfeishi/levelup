import torch
import os
import random
import numpy as np
import cv2
import PIL.Image as Image
import torch.utils.data as data
from my_transforms import preprocess_input, cvtColor

"""
/projection/
    +----/DUTS-TR/
            +-----/DUTS-TR-Image/
            +-----/DUTS-TR-Mask/
    +----/DUTS-TE/
            +-----/DUTS-TE-Image/
            +-----/DUTS-TE-Mask/
    +----/utils/ 
            +-----data_dataset.py  class: my_dataset
            +-----data_preprocess.py  data-preprocess,
            +-----utils.py      some-config, show-text-log-
            +-----utils_metrics   criterion
            +-----train_and_eval.py     criterion      evaluate     fit_one_epoch()
            +-----distributed_utils.py   # windows may no use
    +----train.py
    
"""

"""
/proj/
    +---/dataset/
            +----/train/
            +----/test/
    +---/utils/
            +----my_dataset_b.py
    +---my_dataset_a.py   path = "../dataset" True
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
        window_size = (640, 640)
        window_w, window_h = window_size
        pixel_num_threshold_min = 4000
        if w > window_w and h > window_h:
            while True:
                x = random.randint(0, w-window_w)
                y = random.randint(0, h-window_h)

                # cv2-image: h, w, c
                jpg_crop = image.crop((x, y, x+window_w, y+window_h))
                png_crop = target.crop((x, y, x+window_w, y+window_h))

                png_crop_array = np.array(png_crop)

                non_background_pixels = np.sum(png_crop_array != 0)
                if non_background_pixels > pixel_num_threshold_min:
                    break

        # img-transforms
        jpg_crop, png_crop = self.my_img_augment(jpg_crop, png_crop, window_size, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True)

        #  ToTensor
        jpg_crop = np.transpose(preprocess_input(np.array(jpg_crop, np.float64)), [2, 0, 1])
        png_crop = np.array(png_crop)  # png_crop: (ww, wh)
        png_crop[png_crop >= self.num_classes] = self.num_classes   # may no use

        # tensor-transforms
        # sth

        #   to  one_hot
        seg_labels = np.eye(self.num_classes+1)[png_crop.reshape([-1])]
        seg_labels = seg_labels.reshape((int(window_h), int(window_w), self.num_classes+1))
        # print(seg_labels.shape)  # (ww, wh, 4)  (2+1 +1)  --> 为了好计算loss

        return jpg_crop, png_crop, seg_labels

    def __len__(self):
        return len(self.images_path)

    def my_img_augment(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        # ------------------------------------------#
        #   高斯模糊
        # ------------------------------------------#
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # ------------------------------------------#
        #   旋转
        # ------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    @staticmethod
    def collate_fn(batch):
        images = []
        pngs = []
        seg_labels = []
        for img, png, labels in batch:
            images.append(img)
            pngs.append(png)
            seg_labels.append(labels)
        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        pngs = torch.from_numpy(np.array(pngs)).long()
        seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
        return images, pngs, seg_labels




if __name__ == '__main__':
    path = "../dataset"    # True
    print(os.path.exists(path))

    train_dataset = MyDataset("../", train=True)
    print(len(train_dataset))

    val_dataset = MyDataset("../", train=False)
    print(len(val_dataset))

    i, t, l = train_dataset[0]