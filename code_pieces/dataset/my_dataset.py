import os
import cv2
import torch.utils.data as data
import numpy as np
import torch
import random

from PIL import Image

class DUTSDataset(data.Dataset):
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
        image_crop, mask_crop = self.pillow_random_crop(image_path, mask_path, window_size=(800, 640))

        # print("aaa in", mask_crop.shape)

        if self.transforms is not None:
            image_crop, mask_crop = self.transforms(image_crop, mask_crop)

        # print("mask_crop shape", mask_crop.shape)

        crop_label = self.rgb_to_class_index(mask_crop, key_rgb_map={'1': (30, 200, 40), '2': (60, 80, 210)})
        print(crop_label.shape)  # opencv: torch.Size([640, 800])   # <transform.ToTensor 处理了> --train.py torch.Size([3, 576])


        print("while loop out")
        crop_label = crop_label.long().squeeze()   # long整数类型，然后压缩掉为1的维度，
        print("aaa", crop_label.shape)  # opencv: aaa torch.Size([640, 800])
        return image_crop, crop_label

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def opencv_random_crop(image_path, mask_path, window_size=(800, 640), target_pix_num_th=8000):
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        assert image is not None, f"failed to read image: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = image.shape

        target = cv2.imread(mask_path, flags=cv2.IMREAD_COLOR)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        assert target is not None, f"failed to read mask: {mask_path}"

        crop_width, crop_height = window_size
        # 确保图像尺寸足够大
        if w >= crop_width and h >= crop_height:
            while True:  # 无限循环  直到遇到下一个brake or  return
                # 随机选择裁剪起点
                x = random.randint(0, w - crop_width)
                y = random.randint(0, h - crop_height)

                # opencv使用切片进行裁剪, 默认是不会处理通道数的， cv2-image h, w, c
                image_crop = image[y:y + crop_height, x:x + crop_width]
                mask_crop = target[y:y + crop_height, x:x + crop_width]

                # 计算非背景像素的数量（假设背景标签为0）
                non_background_pixels = np.sum(np.any(mask_crop != 0, axis=2))

                # 检查非背景像素是否满足要求
                if non_background_pixels > target_pix_num_th:
                    return image_crop, mask_crop

    @staticmethod
    def pillow_random_crop(image_path, mask_path, window_size=(800, 640), target_pix_num_th=8000):
        image = Image.open(image_path)
        assert image is not None, f"failed to read image: {image_path}"

        mask = Image.open(mask_path)
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
                mask_crop = mask.crop((x, y, x + crop_width, y + crop_height))

                # 将PNG裁剪图像的像素值转换为 numpy 数组以方便处理
                mask_array = np.array(mask_crop)

                # print("mask crop", mask_crop)

                # 计算非背景像素的数量（假设背景标签为0）
                non_background_pixels = np.sum(mask_array != 0)

                # 检查非背景像素是否满足要求
                if non_background_pixels > target_pix_num_th:
                    return image_crop, mask_crop


    @staticmethod
    def rgb_to_class_index(mask, key_rgb_map):
        # 如果是pillow就需要这一步转换， opencv就是 numpy.ndarray
        mask = np.array(mask)

        # Create an empty class index map
        print("opencv: mask in shape",mask.shape)  # opencv: mask in shape (640, 800, 3)  # --train.py opencv: mask in shape torch.Size([3, 576, 576])
        print("opencv: mask in type:", type(mask))  # opencv: mask in type: <class 'numpy.ndarray'> # train.py opencv: mask in type: <class 'torch.Tensor'>
        class_index_map = torch.zeros(mask.shape[:2], dtype=torch.long)   # 构建一个 torch.tensor
        print("opencv: mask-map size", mask.shape)  # opencv: mask-map size (640, 800, 3) # --train.py opencv: mask-map size torch.Size([3, 576, 576])
        print("opencv: mask-map type", type(class_index_map))  # opencv: mask-map type <class 'torch.Tensor'>
        print("opencv: class idx map", class_index_map.size())  # opencv: class idx map torch.Size([640, 800])
        # Map each color to its corresponding class ID
        for class_id, rgb_values in key_rgb_map.items():
            matches = np.all(mask == np.array(rgb_values), axis=-1)  # axis=-1 对其最内层维度(颜色通道) 比较。-- h,w,c 的c通道嘛
            class_index_map[matches] = int(class_id)
        print("opencv: mask-map shape", class_index_map.shape)  # opencv: mask-map shape torch.Size([640, 800])
        return class_index_map

    # @staticmethod
    # def pillow_rgb_to_class_index(mask, key_rgb_map):
    #     print("mask-in type", type(mask))  # mask-in type <class 'PIL.Image.Image'>   # train.py中为什么就变成torch.tensor了
    #     print("mask-in size ", mask.size)  # mask-in size  (800, 640)
    #     # PIL.Image  --> array
    #     mask = np.array(mask)
    #     # Create an empty class index map
    #     print("pillow: mask in type", type(mask))  # pillow: mask in type <class 'numpy.ndarray'>
    #     print("pillow: mask-in size", mask.size)  # pillow: mask-in size 1536000
    #     class_index_map = torch.zeros(mask.shape[:2], dtype=torch.long)
    #     print("pillow: mask-map size", class_index_map.size())  # pillow: mask-map size torch.Size([640, 800])
    #     print("pillow: mask-map shape", class_index_map.shape)  # pillow: mask-map shape torch.Size([640, 800])
    #
    #     # Map each color to its corresponding class ID
    #     for class_id, rgb_values in key_rgb_map.items():
    #         matches = np.all(mask == np.array(rgb_values), axis=-1)
    #         class_index_map[matches] = int(class_id)
    #     # print(class_index_map)
    #     return class_index_map


    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack([torch.from_numpy(np.array(img)) for img in images])
        labels = torch.stack([torch.from_numpy(np.array(lbl)) for lbl in labels])
        return images, labels


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs




if __name__ == '__main__':
    train_dataset = DUTSDataset("./", train=True)
    print(len(train_dataset))

    val_dataset = DUTSDataset("./", train=False)
    print(len(val_dataset))

    i, t = train_dataset[0]
