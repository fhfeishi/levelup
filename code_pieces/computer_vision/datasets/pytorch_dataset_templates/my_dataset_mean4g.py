import os
import cv2
import numpy as np
import torch.utils.data as data
import torch

import warnings
warnings.simplefilter('ignore')

class DUTSDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None, test_crop=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        self.image_root = os.path.join(root, "DUTS-TR" if train else "DUTS-TE",
                                       "DUTS-TR-Image" if train else "DUTS-TE-Image")
        self.mask_root = os.path.join(root, "DUTS-TR" if train else "DUTS-TE",
                                      "DUTS-TR-Mask" if train else "DUTS-TE-Mask")
        assert os.path.exists(self.image_root) and os.path.exists(self.mask_root), "Image or mask path does not exist."

        self.image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
        self.mask_names = [p.replace(".jpg", ".png") for p in self.image_names]
        assert all(os.path.exists(os.path.join(self.mask_root, m)) for m in
                   self.mask_names), "Some images do not have corresponding masks."

        self.transforms = transforms

        # 是否测试一下dataset写没写错
        self.test_crop = test_crop

    def __len__(self):
        return len(self.image_names) * 4

    def __getitem__(self, idx):

        img_idx = idx // 4   # Determine which image this index corresponds to
        block_idx = idx % 4  # Determine which block of the image to fetch

        image_path = os.path.join(self.image_root, self.image_names[img_idx])
        mask_path = os.path.join(self.mask_root, self.mask_names[img_idx])

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        h, w, _ = image.shape
        block_height = h // 2
        block_width = w // 2

        # Determine the coordinates of the block
        row_idx = block_idx // 2
        col_idx = block_idx % 2
        y = row_idx * block_height
        x = col_idx * block_width

        image_block = image[y:y + block_height, x:x + block_width]
        mask_block = mask[y:y + block_height, x:x + block_width]

        if self.transforms:
            image_block, mask_block = self.transforms(image_block, mask_block)

        if self.test_crop is not None:
            return image_block, mask_block, (x, y, x+block_width, y+block_height)
        else:
            return image_block, mask_block

    @staticmethod
    def collate_fn(batch):
        images, masks = zip(*batch)
        # Convert to tensors or perform other batch aggregation here
        return torch.stack(images, 0), torch.stack(masks, 0)

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
    num = 15
    image_path = os.path.join(train_dataset.image_root, train_dataset.image_names[num])
    mask_path = os.path.join(train_dataset.mask_root, train_dataset.mask_names[num])

    original_image = Image.open(image_path).convert('RGB')
    original_mask = Image.open(mask_path).convert('RGB')

    draw_image = ImageDraw.Draw(original_image)
    draw_mask = ImageDraw.Draw(original_mask)

    colors = ["red", "yellow", "blue", "green"]

    for i in range(4):
        _, _, bbox = train_dataset[i]
        draw_image.rectangle(bbox, outline=colors[i], width=5)
        draw_mask.rectangle(bbox, outline=colors[i], width=5)

    original_image.show()
    original_mask.show()

