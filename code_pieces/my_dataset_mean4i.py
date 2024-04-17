import os
import cv2
import numpy as np
import torch
import torch.utils.data as data

import warnings

warnings.simplefilter('ignore')

class DUTSDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None, test_crop=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_root = os.path.join(root, "DUTS-TR", "DUTS-TR-Image")
            self.mask_root = os.path.join(root, "DUTS-TR", "DUTS-TR-Mask")
        else:
            self.image_root = os.path.join(root, "DUTS-TE", "DUTS-TE-Image")
            self.mask_root = os.path.join(root, "DUTS-TE", "DUTS-TE-Mask")
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

        # test mean strategy
        self.test_crop = test_crop

        # initialize blocks information
        self.blocks = []
        for img_idx, image_n in enumerate(image_names):
            mask_n = image_n.replace('.jpg', '.png')
            mask_p = os.path.join(self.mask_root, mask_n)
            image_p = os.path.join(self.image_root, image_n)
            image = cv2.imread(image_p)
            h, w, _ = image.shape
            block_height = h//2
            block_width = w//2

            # create entries for each block
            for block_idx in range(4):
                row_idx = block_idx // 2
                col_idx = block_idx % 2
                y = row_idx * block_height
                x = col_idx * block_width
                self.blocks.append((image_p, mask_p, x, y, block_width, block_height))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):

        image_path, mask_path, x, y, block_width, block_height = self.blocks[idx]

        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        assert image is not None, f"failed to read image: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = image.shape

        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"failed to read mask: {mask_path}"

        image_block = image[y:y+block_height, x:x+block_width]
        mask_block = mask[y:y+block_height, x:x+block_width]

        if self.transforms is not None:
            image_block, mask_block = self.transforms(image_block, mask_block)

        if self.test_crop is not None:
            return image_block, mask_block, (x, y, x + block_width, y + block_height)
        else:
            return image_block, mask_block

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
    from PIL import Image, ImageDraw
    import transforms as T
    from typing import Union, List

    class SODPresetTrain:
        def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                     hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Resize(base_size, resize_mask=True),
                T.RandomCrop(crop_size),
                T.RandomHorizontalFlip(hflip_prob),
                T.Normalize(mean=mean, std=std)
            ])
            print('----resized!!!!!!')

        def __call__(self, img, target):
            return self.transforms(img, target)

    class SODPresetEval:
        def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Resize(base_size, resize_mask=False),
                T.Normalize(mean=mean, std=std),
            ])

        def __call__(self, img, target):
            return self.transforms(img, target)

    train_dataset = DUTSDataset('./', train=True, test_crop=True, transforms=SODPresetTrain([1024, 800], crop_size=576))
    print(len(train_dataset))

    test_dataset = DUTSDataset('./', train=False)
    print(len(test_dataset))

    print(train_dataset.blocks[0])
    print(train_dataset.blocks[1])
    print(train_dataset.blocks[2])
    print(train_dataset.blocks[3])



    # num = 2
    # image_path = os.path.join(train_dataset.image_root, train_dataset.blocks[0][num])
    # mask_path = os.path.join(train_dataset.mask_root, train_dataset.blocks[0][num])

    # original_image = Image.open(image_path).convert('RGB')
    # original_mask = Image.open(mask_path).convert('RGB')
    #
    # draw_image = ImageDraw.Draw(original_image)
    # draw_mask = ImageDraw.Draw(original_mask)
    #
    # colors = ["red", "yellow", "blue", "green"]
    #
    # for i in range(4):
    #     _, _, bbox = train_dataset[i]
    #     draw_image.rectangle(bbox, outline=colors[i], width=5)
    #     draw_mask.rectangle(bbox, outline=colors[i], width=5)
    #
    # original_image.show()
    # original_mask.show()


