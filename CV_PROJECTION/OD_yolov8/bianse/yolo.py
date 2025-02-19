import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox, mergeSlide_results
from collections import OrderedDict, defaultdict
from slide import imageSlide

class YOLO(object):
    _defaults = {

        "model_path"        : 'logs/model_0826_1.pth',
        "classes_path"      : 'model_data/my_classes.txt',
        "input_shape"       : [640, 640],
        "phi"               : 's',
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "letterbox_image"   : True,
        "cuda"              : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.bbox_util                      = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    def generate(self, onnx=False):
        
        #   建立yolo模型，载入yolo模型的权重
        self.net    = YoloBody(self.input_shape, self.num_classes, self.phi)
        # device： single_gpu or cpu
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.net.load_state_dict(new_state_dict)
        
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, crop = False, count = False):
        
        #   计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image       = cvtColor(image)
        
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            #   将图像输入网络当中进行预测！
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # outputs: tensor(
            # torch.Size([1,4,8400]), 
            # torch.Size([1,6,8400]), 
            # (torch.Size([1，7，80，80])、torch.Size([1，7，40，40])、torch.Size([1，7，20，20])), 
            # torch.Size([2，8400], 
            # torch.Size([1，8400]
            # )
            # dbox, cls, origin_cls, anchors, strides
            
            #   将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
            
            # nms cross class --->>> may one_region has two pred_bbox or more
            top_label,top_conf, top_boxes  = self.nms_cross_class(top_label, 
                                                                top_conf, 
                                                                top_boxes,
                                                                iou_threshold=0.85)
            
        
        #   设置字体与边框厚度
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        
        #   计数
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        
        #   是否进行目标的裁剪
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        
        #   图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def detec_slideImage(self, image):
        image       = cvtColor(image)  # input_image
        image_data  = resize_image(image, (1280, 960), self.letterbox_image)
        imageBlocks = imageSlide()
        imageBlocksDict = imageBlocks.paddedImage_crop(image_data, crop_size=640, stride=(320,320))
        
        results_list = []
        for rc_idx, iblock in imageBlocksDict.items():
            image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(iblock, dtype='float32')), (2, 0, 1)), 0)
            row_idx, col_idx = rc_idx
            with torch.no_grad():
                images = torch.from_numpy(image_data)
                if self.cuda:
                    images = images.cuda()
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        self.input_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
                if results[0] is None: 
                    continue
                
                top_label   = np.array(results[0][:, 5], dtype = 'int32')
                top_conf    = results[0][:, 4]
                top_boxes   = results[0][:, :4]
                
                # merge silides-detec_results
                top_boxes[:,0] += imageBlocks.stride_height * row_idx  # top y1
                top_boxes[:,1] += imageBlocks.stride_width * col_idx  # left x1
                top_boxes[:,2] += imageBlocks.stride_height * row_idx  # bottom y2
                top_boxes[:,3] += imageBlocks.stride_width * col_idx  # right x2
                for i in range(len(top_label)):
                    results_list.append([top_label[i], top_conf[i], top_boxes[i]])
                    
        # merge same top_label  predict 
        # Merge the results
        merged_results = mergeSlide_results(results_list) # xms in class
        # mergeSlide_results返回  ndarray：label, score, box       
        #                       python会默认封装元组然后赋值给merged_results
        # return _predictClass _confidence _predictBbox   --> image
        
        if merged_results is not None:
            
            # nms cross class --->>> may one_region has two pred_bbox or more
            merged_results = self.nms_cross_class(merged_results[0], 
                                                  merged_results[1], 
                                                  merged_results[2],
                                                  iou_threshold=0.85)
            
            # draw merged_result on image
            image_new = image.copy()
            font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
                    
            for label, score, box in merged_results:
                class_id = int(label)
                predicted_class = self.class_names[class_id]
                top, left, bottom, right = box

                # Ensure box coordinates are within image boundaries
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                # Create the label string
                label_text = f'{predicted_class} {score:.2f}'
                draw = ImageDraw.Draw(image_new)
                label_size = draw.textsize(label_text, font)
                label = label_text.encode('utf-8')
                print(label_text, top, left, bottom, right)

                # Determine the text origin position
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # Draw the bounding box with the specified thickness
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[class_id])

                # Draw the label background and text
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[class_id])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                
                # Clean up the drawing object
                del draw
        else:
            return image
        return image_new      
    
    def nms_cross_class(self, top_labels, top_confs, top_boxes, iou_threshold=0.85):
        # 结果列表
        nms_indices = []
        # 合并top_labels, top_confs和top_boxes以便于处理
        boxes_confs_labels = np.concatenate([top_boxes, top_confs[:, None], top_labels[:, None]], axis=1)
        
        # 按置信度对所有边界框进行排序（降序）
        sorted_indices = np.argsort(-boxes_confs_labels[:, 4])
        sorted_boxes_confs_labels = boxes_confs_labels[sorted_indices]
        
        while len(sorted_boxes_confs_labels) > 0:
            # 选取置信度最高的边界框
            current_box = sorted_boxes_confs_labels[0]  # .pop(0) 取出来 ，这是对np
            # 添加到结果列表中
            nms_indices.append(sorted_indices[0])
            if len(sorted_boxes_confs_labels) == 1:
                break
            
            # 计算选中边界框与其他所有边界框的IoU
            ious = self.compute_iou(current_box[np.newaxis, :4], sorted_boxes_confs_labels[1:, :4])
            
            # 保留IoU小于阈值的边界框
            remaining_indices = np.where(ious < iou_threshold)[0]  # np.ndarray --> list_index
            
            # 更新待处理的边界框和索引
            sorted_boxes_confs_labels = sorted_boxes_confs_labels[remaining_indices + 1]  # +1 因为要跳过已选择的边界框
            sorted_indices = sorted_indices[remaining_indices + 1]  # 更新排序后的索引以保持一致
        
        # 根据保留的索引提取结果
        top_labels = top_labels[nms_indices]
        top_confs = top_confs[nms_indices]
        top_boxes = top_boxes[nms_indices]

        return top_labels, top_confs, top_boxes
        
    def compute_iou(self, box, boxes):
        # np.array: box boxes
        
        # inter arae
        x1 = np.maximum(box[:, 0], boxes[:, 0])
        y1 = np.maximum(box[:, 1], boxes[:, 1])
        x2 = np.minimum(box[:, 2], boxes[:, 2])
        y2 = np.minimum(box[:, 3], boxes[:, 3])
        intersection = np.maximum(0, (x2-x1)) * np.maximum(0, (y2-y1))
        
        box_area = (box[:2]-box[:,0]) * (box[:,3]-box[:,1])
        boxes_area = (boxes[:2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        
        union = box_area + boxes_area - intersection
        
        return intersection/union
         
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])   # 这里的image是什么类型的数据？
        
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image       = cvtColor(image)
        
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        
        #   添加上batch_size维度
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            #   将图像输入网络当中进行预测！
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            
            #   将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

            # one bbox two pred_results ??    nms cross class
            top_label, top_conf, top_boxes = self.nms_cross_class(top_label, top_conf, top_boxes, iou_threshold=0.85)
            
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return
        
