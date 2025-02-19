import torch
import numpy as np
import cv2
import os
import random
import copy


class MyData(torch.utils.data.Dataset):
    def __init__(self, cfg, training=False):
        self.root = cfg['root']
        self.crop_size = cfg['crop_size']

        self.files = [i[:-4] for i in os.listdir(os.path.join(self.root, 'aug')) if i.endswith('.jpg')]
        self.training = training

        if training:
            random.shuffle(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        name = self.files[index]
        if self.training:

            t = 'aug'
            img_name = os.path.join(self.root, t, name+'.jpg')
            gt_name = os.path.join(self.root, t, name+'.png')
        else:
            img_name = os.path.join(self.root, 'data7', name+'.jpg')
            gt_name = os.path.join(self.root, 'data7', name+'.png')
        img = cv2.imread(img_name)
        gt = cv2.imread(gt_name)
        return self.transform(img, gt, name)

    def transform(self, img, gt, name):

        img = cv2.resize(img, (512,352), cv2.INTER_CUBIC)
        gt = cv2.resize(gt, (512,352), cv2.INTER_NEAREST)

        gt = np.array(gt, dtype=np.float32)
        gt /= 255

        img = np.array(img, dtype=np.float32)
        img = img/255
        return 