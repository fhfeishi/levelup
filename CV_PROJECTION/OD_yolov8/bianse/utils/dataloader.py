import torch
import random 
import cv2
from random import sample, shuffle
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input

class YoloDataset(Dataset):
    def __init__(self, annotation_lines,
                 model_inShape,
                 num_classes,
                 epoch_length,
                 mosaic, mosaic_prob,
                 mixup, mixup_prob,
                 train,
                 special_aug_ratio):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.model_inShape = model_inShape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_raio = special_aug_ratio
        
        self.epoch_now = -1
        self.length = len(self.annotation_lines)
        self.bbox_attrs = 5+ num_classes
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        
        index = index%self.length

        # train aug not val
        if self.mosaic and self.rand()<self.mosaic \
        and self.epoch_now<(self.special_aug_raio*self.epoch_length):
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box = self.get_random_data_with_mosaic(lines, 
                                    self.model_inShape)
            
            if self.mixup and self.rand()<self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_ramdom_data(lines[0], 
                                    self.model_inShape, random_=self.train)
                image, box = self.get_random_data_with_Mixup(
                                image, box, image_2, box_2)
        else:
            image, box = self.get_random_data(self.annotation_lines[index],
                                self.model_inShape, random_=self.train)
            
        image = np.transpose(preprocess_input(np.array(image,
                                    dtype=np.float32)), (2,0,1))
        box = np.array(box, dtype=np.float32)
        
        # preprocess gt-bbox
        nL = len(box)
        lables_out = np.zeros((nL, 6))
        if nL:
            # box: x1 y1 x2 y2 cls-id
            box[:, [0,2]] /= self.model_inShape[1]
            box[:, [1,3]] /= self.model_inShape[0]
            # reshape to: x y w h cla-id
            box[:, 2:4] -= box[:, 0:2]
            box[:, 0:2] += (box[:, 2:4]/2)
            # reshape to: 0(conf) cla-id x y w h
            lables_out[:, 1] = box[:, -1]  # cls-id  ## labels_out[:, 0]
            lables_out[:, 2:] = box[:, :4] #        x y w h
        return image, lables_out
    
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def get_random_data(self, annotation_line, model_inShape,
            jitter=.3, hue=.3, sat=.7, val=.7, random_=True):
        # annotation_line -> txtFile.readlines() - path x1 y1 x2 y2 cla-id
        line = annotation_line.split()
        image = Image.open(line[0]).convert('RGB')
        
        iw, ih = image.size
        w, h = model_inShape
        
        box = np.array([np.array(list(map(int, box.split(',')))) 
                        for box in line[1:]])
        
        if not random_:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2  # // 取商  %取余 / 除法
            dy = (h-nw)//2
            
        # 四周补灰条
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)
        
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            #  # discard invalid box
            box_w = box[:,2]-box[:,0]
            box_h = box[:,3]-box[:,1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        return image_data, box
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                if i == 0:
                    if y1>cuty or x1>cutx:
                        continue
                    if y2>=cuty and y1<=cuty:
                        y2=cuty
                    if x2>=cutx and x1<=cutx:
                        x2=cutx
    
    def get_random_data_with_Mosaic(self, annotation_line,
            model_inShape, jitter=.3, hue=.4, sat=.7, val=.4):
        h, w = model_inShape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)      
        
        image_datas = []
        box_datas = []
        index = 0
        # 4 images_
        shuffle(annotation_line)
        for line in annotation_line:
            line_content = line.split()
            image = Image.open(line_content[0]).convert('RGB')
            iw, ih = image.size
            box = np.array([np.array(list(map(int,box.split(',')))) 
                            for box in line_content[1:]])
            
            LeftRight_flip = self.rand() < 0.5
            if LeftRight_flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, 0,2] = iw - box[:, [2,0]]

            # resize h->nh scale1, w->nw scale2
            new_ar = iw/ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
            scale = self.rand(0.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            
            # 4 images expand
            # 0 3
            # 1 2
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x) 
                dy = int(h*min_offset_y) 
            else:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx,dy))
            image_data = np.array(new_image)
            
            index += 1
            box_data = []
            
            # box process
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)
            
        # cut_cat
        # 0 3
        # 1 2
        cutx = int(w*min_offset_x)
        cuty = int(h*min_offset_y)
        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
        new_image       = np.array(new_image, np.uint8)
        # color transform
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
        # merge box
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes
            
    def get_random_data_with_Mixup(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes == box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes
        
    def randomCropBbox(self, pillowImage, box, crop_size=(640, 640), area_thr=0.5, max_attempts=2000):
        # only for train------------------
        # box: [[x1, y1, x2, y2, cla-id], ...]
        w, h = pillowImage.size
        assert w>=crop_size[0] and h>=crop_size[1], "crop < image!!!!!"
        
        for _ in range(max_attempts):
            x1 = random.randint(0, (w-crop_size[0]+1))
            y1 = random.randint(0, (h-crop_size[1]+1))
            x2 = x1 + crop_size[0]
            y2 = y1 + crop_size[1]

            box_dict = {}
            for b in box:
                inter_x1 = max(b[0], x1)
                inter_y1 = max(b[1], y1)
                inter_x2 = min(b[2], x2)
                inter_y2 = min(b[3], y2)
                
                b_iou = (max(0,inter_x2-inter_x1)*(max(0,inter_y2-inter_y1)))/((b[2]-b[0])*(b[3]-b[1]))
                box_dict[tuple(b)] = b_iou
                
            if any(b_i >= area_thr for b_i in box_dict.values()) and \
                all(b_i >= area_thr or b_i == 0 for b_i in box_dict.values()):
                    croped_image = pillowImage.crop((x1,y1,x2,y2))
                    # update bbox
                    new_bboxes = []
                    for bbox, ratio in box_dict.items():
                        if ratio > area_thr:
                            updated_bbox = [
                                        max(0, bbox[0] - x1),
                                        max(0, bbox[1] - y1),
                                        min(crop_size[0], bbox[2] - x1),
                                        min(crop_size[1], bbox[3] - y1)]
                            new_bboxes.append(updated_bbox)
                    
                    return croped_image, new_bboxes
                
        return pillowImage, box
            
            