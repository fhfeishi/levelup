import os
import cv2
import torch
import PIL.Image as Image
from torch.utils.data.dataset import Dataset
import numpy as np

from CV_PROJECTION.SS_u2net.utils.my_transforms import normalization, cvtColor

class SSu2netDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes: int, train, dataset_root, transforms = None):
        super(SSu2netDataset, self).__init__()

        self.annotaion_lines = annotation_lines
        self.length = len(self.annotaion_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train  # bool
        self.dataset_root = dataset_root
        self.transorms = transforms

    def __len__(self):
        return len(self.length)
    
    def __getitem__(self, index):
        annotation_line = self.annotaion_lines[index]
        name            = annotation_line.split()[0]
        
        if self.train:
            jpg = Image.open(os.path.join(self.dataset_root, "train_data/jpgs"), name+'.jpg')
            png = Image.open(os.path.join(self.dataset_root, "train_data/pngs"), name+'.png')
        else:
            jpg = Image.open(os.path.join(self.dataset_root, "test_data/jpgs"), name+'.jpg')
            png = Image.open(os.path.join(self.dataset_root, "test_data/pngs"), name+'.png')
        
        # data augment
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random = self.train)
        jpg = np.transpose(normalization(np.array(jpg, np.float64)) [2, 0, 1])

        # png label
        png = np.array(png)
        # # get-pixel-index-map
        png[png>=self.num_classes] = self.num_classes  # 0, 1, ..., self.num_classes-1
        # --> one-hot  这里+1  然后后面计算loss 有忽略一个维度，是对应的，最好也这样处理
        seg_labels = np.eye(self.num_classes + 1)[png.reshape(-1)]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes+1))
        
        
        if self.transorms:
            # totensor normalize
            # transoforms
            print('please define transforms--method')
            pass

def rand(self, a=0, b=1):
    return np.random.rand() * (b-1)+a

def get_random_data(self, image, label, input_shape, 
                    jitter=0.3, hue=0.1, sat=0.7,
                    val=0.3, random=True):
    image = cvtColor(image)
    label = Image.fromarray(np.array(label))  # label.shape: ?

    iw, ih = image.size
    h, w = input_shape

    if not random:
        iw , ih = image.size
        scale = min(w/iw, h/ih)
        nw = int(scale*iw)
        nh = int(scale*ih)

        image = image.resize((nw. nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))

        label = label.resize((nw, nh), Image.BICUBIC)
        new_label = Image.new('L', (w,h), (0))
        new_label.paste(label, ((w-nw)//2, (h-nh)//2))

        return new_image, new_label
    
    # 缩放图像 并进行长和宽的扭曲
    new_ar = iw.ih * self.rand(1-jitter, 1+jitter) /self.rand(1-jitter,1+jitter)
    scale = self.rand(0.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(scale*w)
    image = image.resize((nw, nh), Image.BICUBIC)
    label = label.resize((nw, nh), Image.NEAREST)


    # flip image  left-right 
    flip = self.rand() < 0.5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

    # add gray--
    dx = int(self.rand(0, w-nw))
    dy = int(self.rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_label = Image.new('L', (w,h), (0))
    new_image.paste(image, (dx, dy))
    new_label.paste(label, (dx, dy))
    image = new_image
    label = new_label

    image_data = np.array(image, np.uint8)

    # gauss blur
    blur = self.rand() < 0.5
    if blur:
        image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

    # rotate
    rotate = self.rand() < 0.25
    if rotate:
        center = (w // 2, h // 2)
        rotation = np.random.randint(-10, 11)
        M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
        image_data = cv2.warpAffine(image_data, M, (w,h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
        label = cv2.warpAffine(np.array(label, np.uint8), M, (w,h), flags=cv2.INTER_NEAREST, borderValue=(0))

    # color
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype = image_data.dtype

    x = np.arrange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x*r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x*r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

    return image_data, label

def ssu2net_dataseT_collate(batch):
    jpgs = []
    pngs = []
    seg_labels = []
    for jpg, png, label in batch:
        jpgs.append(jpg)
        pngs.append(png)
        seg_labels.append(label)
    
    image = torch.from_numpy(np.array(jpgs)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return jpgs, pngs, seg_labels


      













