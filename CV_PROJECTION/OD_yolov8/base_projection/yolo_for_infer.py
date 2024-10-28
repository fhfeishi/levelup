import colorsys
import os
import numpy as np
import torch
import torch.nn as nn
import PIL
import cv2
from PIL import ImageDraw, ImageFont
from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox, check_version
from collections import OrderedDict, defaultdict


# read config.yaml
import yaml
with open("cfg.yaml", "r") as file:
    cfg = yaml.safe_load(file)


# #  ----------------------  slide  -------------------------  #
# option1: crop_size=(a,b) and row_num=2 col_num=2
# option2: crop_size=(a,b) and stride=(c,d)
# imageBlockType_rgb: opencv pillow
class SlideImage(object):
    def __init__(self):

        # 输入图片的高度宽度
        self.w = None
        self.h = None

        # 缩放之后，图片的高度、宽度
        self.scaled_h = None
        self.scaled_w = None

        # 滑窗大小
        self.crop_size_w = None
        self.crop_size_h = None

        # 滑窗移动步长：宽度方向 高度方向
        self.stride_w = None
        self.stride_h = None

        # 对原图下方补零长度 右方补零长度
        self.padding_w = None
        self.padding_h = None
        self.padding_bottom = None
        self.padding_right = None
        self.padding_top = None
        self.padding_left = None

        # 滑窗的移动步数宽度/高度方向 、滑窗截图的行/列数
        self.cols_num = None   # h
        self.rows_num = None   # w

        # image_block row_index col_index
        self._ImageBlock = {}  # key: (row_index, col_index), value:image_block

    def slide_image(self, image, crop_size, outImageMode="pillow", rows_num=None, cols_num=None, stride=None,
                    fill=128, drawBoxForTest=False):

        # check outImageType
        assert outImageMode in ['pillow', 'opencv'], "outImageType must be 'pillow' or 'opencv'"
        # import --
        if outImageMode == "opencv":
            import cv2
        else:
            from PIL import Image

        # if 'pillow':  import PIL.Image, ImageOps
        if outImageMode == "pillow":
            from PIL import Image, ImageDraw, ImageFont, ImageOps

        # #------check input image---------#
        # pillowImage or opencvImage
        if isinstance(image, np.ndarray) and image.shape[2] ==3:
            # opencv image  ch-3
            self.h, self.w, _ = image.shape
            if outImageMode == "pillow":
                image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            # pillow image ch-3
            self.w, self.h = image.size
            if outImageMode == "opencv":
                image = np.array(image)
        else:
            print("wrong input_image type! return None")
            return None

        # #------------check params------------#
        if (rows_num is None or cols_num is None) and stride is None:
            print("define: stride=(a,b) or define: cols_num=intx cols_num=inty")
            print("return None")
            return None

        # crop_size
        self.crop_size_w, self.crop_size_h = crop_size
        # stride
        self.stride_w, self.stride_h = stride if stride is not None else (None, None)

        if rows_num is None and cols_num is None:
            # rows num
            rest_h = self.h - crop_size[1]
            if rest_h % self.h != 0:
                num_rows = int(rest_h // self.stride_h) + 1
                self.rows_num = num_rows + 1
            else:
                num_rows = int(rest_h / self.stride_h)
                self.rows_num = num_rows + 1

            # cols num
            rest_w = self.w - crop_size[0]
            if rest_w % self.stride_w != 0:
                num_cols = int(rest_w // self.stride_w) + 1
                self.cols_num = num_cols + 1
            else:
                num_cols = int(rest_w // self.stride_w)
                self.cols_num = num_cols + 1

        else:
            assert cols_num > 1 and rows_num > 1, "cols_num, rows_num must > 1"
            self.rows_num, self.cols_num = rows_num, rows_num
            # no padding
            self.stride_w = (self.w-self.crop_size_w)//(self.cols_num-1)
            self.stride_h = (self.h-self.crop_size_h)//(self.rows_num-1)

        # padding height, padding width
        self.padding_h = self.rows_num * self.stride_h + crop_size[1] - self.h
        self.padding_w = self.cols_num * self.stride_w + crop_size[0] - self.w
        self.padding_h = int(self.padding_h) + (int(self.padding_h) % 2)
        self.padding_w = int(self.padding_w) + (int(self.padding_w) % 2)

        # padding crop
        if outImageMode == "pillow":
            # # ---------- padding ------------ #
            # 右边和下面补零  左上右下
            padded_image = ImageOps.expand(image, border=(
                0, 0, self.padding_w, self.padding_h), fill=(fill, fill, fill))
            self.padding_left,self.padding_top, self.padding_right, self.padding_bottom = 0, 0, self.padding_w, self.padding_h

            # # ------------ crop -------------- #  pillow draw ++
            for r in range(self.rows_num):
                for c in range(self.cols_num):
                    start_row = r * self.stride_h
                    end_row = start_row + crop_size[1]
                    start_col = c * self.stride_w
                    end_col = start_col + crop_size[0]
                    block = padded_image.crop((start_col, start_row, end_col, end_row))
                    if block.size[0] == crop_size[0] and block.size[1] == crop_size[1]:
                        self._ImageBlock[(r, c)] = block
        else:
            # # opencv
            # # ---------- padding ------------ #
            # 右边和下面补零  上下左右
            padded_image = cv2.copyMakeBorder(image, 0, self.padding_w, 0, self.padding_h,
                                              cv2.BORDER_CONSTANT, value=(fill,fill,fill))

            # # ------------ crop -------------- #
            for r in range(self.rows_num):
                for c in range(self.cols_num):
                    start_row = r * self.stride_h
                    end_row = start_row + crop_size[1]
                    start_col = c * self.stride_w
                    end_col = start_col + crop_size[0]
                    block = padded_image[start_col:end_col,start_col:end_row]
                    if block.shape[1] == crop_size[0] and block.shape[0] == crop_size[1]:
                        self._ImageBlock[(r, c)] = block

        return self._ImageBlock

# merge slide_image_input_model_out
def merge_overlapped_ltbrBbox(results_list, iou_thre=0.3):
    if results_list is None:
        return None

    class_dict = defaultdict(list)
    for label, conf, bbox in results_list:
        class_dict[label].append((conf, bbox))

    final_results = []

    for label, bboxes in class_dict.items():
        bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)

        while bboxes:
            best_conf, best_bbox = bboxes.pop(0)
            merged_bbox = best_bbox.copy()
            to_merge = []

            for conf, bbox in bboxes:
                if calculate_tlbrIou(best_bbox, bbox) > iou_thre:
                    to_merge.append((conf, bbox))

            for conf, bbox in to_merge:
                merged_bbox = merge_tlbrBox(merged_bbox, bbox)
                best_conf = max(best_conf, conf)
                bboxes.remove((conf, bbox))

            final_results.append([label, best_conf, merged_bbox])

        return final_results
def calculate_tlbrIou(boxA, boxB):  # y1 x1 y2 x2
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
def merge_tlbrBox(boxA, boxB):
    y1 = min(boxA[0], boxB[0])
    x1 = min(boxA[1], boxB[1])
    y2 = max(boxA[2], boxB[2])
    x2 = max(boxA[3], boxB[3])

    return np.array([y1, x1, y2, x2])


class YOLO(object):
    _defaults = {
        "model_path"        : cfg['infer_cfg']['model_path'],
        "classes_path"      : cfg['infer_cfg']['classes_file'],
        "input_shape"       : cfg['model_cfg']['input_shape'],
        "phi"               : cfg['model_cfg']['phi'],
        "confidence"        : cfg['infer_cfg']['confidence'],
        "nms_iou"           : cfg['infer_cfg']['nms_iou'],
        "letterbox_image"   : cfg['infer_cfg']['letterbox_image'],
        "cuda"              : cfg['infer_cfg']['cuda']}

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

        self.net    = YoloBody(self.input_shape, self.num_classes, self.phi)
        
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

        image_shape = (image.size[1], image.size[0])
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
            if results[0] is None:
                return image

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

            # top_label, top_conf, top_boxes = self.nms_cross_class(top_label, top_conf, top_boxes, iou_threshold=0.8)
            
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
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
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]  # 有可能会预测错误，就不用了
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            # # image classification
            # img_crop_ = image.crop((left, top, right, bottom))
            # img_crop_ = predBboxCrop_transform(img_crop_)
            # predict_cla = class_predBboxCrop(img_crop_)
            # label_text = '{} {:.2f}'.format(predict_cla, score)
            
            label_text = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label_text, font)
            label = label_text.encode('utf-8')
            print(label_text, top, left, bottom, right)
            
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

    def dectect_pilimage(self, image):
        image_shape = (image.size[1], image.size[0])
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape,
                image_shape, self.letterbox_image,
                conf_thres=self.confidence, nms_thres=self.nms_iou
            )

            if results[0] is None:
                return image, None, None, None

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

            top_label, top_conf, top_boxes = self.nms_cross_class(top_label, top_conf, top_boxes, iou_threshold=0.8)

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label_text = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = (
                draw.textbbox((0, 0), label_text, font=font)[2] - draw.textbbox((0, 0), label_text, font=font)[0], \
                draw.textbbox((0, 0), label_text, font=font)[3] - draw.textbbox((0, 0), label_text, font=font)[1]) \
                if check_version(PIL.__version__, '9.2.0') else draw.textsize(label_text, font)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label_text, fill=(0, 0, 0), font=font)
            del draw

        return image, top_boxes, top_conf, top_label

    def detect_cvimage(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)  # 自定义函数
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape,
                image_shape, self.letterbox_image,
                conf_thres=self.confidence, nms_thres=self.nms_iou
            )

            if results[0] is None:
                return image, None, None, None

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

            top_label, top_conf, top_boxes = self.nms_cross_class(top_label, top_conf, top_boxes, iou_threshold=0.8)

        # 设置字体和线条粗细
        font_scale = 0.5
        thickness = int(max((image.shape[0] + image.shape[1]) // np.mean(self.input_shape), 1))

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom).astype('int32'))
            right = min(image.shape[1], np.floor(right).astype('int32'))

            label_text = '{} {:.2f}'.format(predicted_class, score)
            # 使用 OpenCV 绘制边框和标签
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_origin = (left, top - label_size[1] if top - label_size[1] > 0 else top + 1)

            # 绘制标签背景和文字
            cv2.rectangle(image, (text_origin[0] - 1, text_origin[1] - label_size[1] - 1),
                          (text_origin[0] + label_size[0] + 1, text_origin[1] + 1), self.colors[c], -1)
            cv2.putText(image, label_text, (text_origin[0], text_origin[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness // 2, lineType=cv2.LINE_AA)

        return image, top_boxes, top_conf, top_label

    def nms_cross_class(self, top_labels, top_confs, top_boxes, iou_threshold=0.85):
        nms_indices = []
        boxes_confs_labels = np.concatenate([top_boxes, top_confs[:, None], top_labels[:, None]], axis=1)
        sorted_indices = np.argsort(-boxes_confs_labels[:, 4])
        sorted_boxes_confs_labels = boxes_confs_labels[sorted_indices]
        while len(sorted_boxes_confs_labels) > 0:
            current_box = sorted_boxes_confs_labels[0]
            nms_indices.append(sorted_indices[0])
            if len(sorted_boxes_confs_labels) == 1:
                break
            ious = self.compute_iou(current_box[np.newaxis, :4], sorted_boxes_confs_labels[1:, :4])
            remaining_indices = np.where(ious < iou_threshold)[0]
            sorted_boxes_confs_labels = sorted_boxes_confs_labels[remaining_indices + 1]
            sorted_indices = sorted_indices[remaining_indices + 1]
        top_labels = top_labels[nms_indices]
        top_confs = top_confs[nms_indices]
        top_boxes = top_boxes[nms_indices]
        return top_labels, top_confs, top_boxes
    
    def compute_iou(self, box, boxes):
        x1 = np.maximum(box[:, 0], boxes[:, 0])
        y1 = np.maximum(box[:, 1], boxes[:, 1])
        x2 = np.minimum(box[:, 2], boxes[:, 2])
        y2 = np.minimum(box[:, 3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        iou = intersection / union
        return iou
        

    def detec_slidepilimage(self, image):
        image  = resize_image(image, (1280, 960), self.letterbox_image)
        imageBlocks = SlideImage()
        imageBlocksDict = imageBlocks.slide_image(image, crop_size=(640,640), stride=(320,320))

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
                
                # print("results:", results)        
                          
                if results[0] is None: 
                    continue
                # results ndarray 
                top_label   = np.array(results[0][:, 5], dtype = 'int32')
                top_conf    = results[0][:, 4]
                top_boxes   = results[0][:, :4]       # top, left, bottom, right    y1,x1 y2,x2
                
                top_boxes[:,0] += imageBlocks.stride_h * row_idx  # y1
                top_boxes[:,1] += imageBlocks.stride_w * col_idx  # x1
                top_boxes[:,2] += imageBlocks.stride_h * row_idx  # y2
                top_boxes[:,3] += imageBlocks.stride_w * col_idx  # x2
                # print(top_boxes)
                for i in range(len(top_label)):
                    results_list.append([top_label[i], top_conf[i], top_boxes[i]])
        # print("results_list:", results_list)
        # merge same top_label  predict 
        # Merge the results
        merged_results = merge_overlapped_ltbrBbox(results_list)
        # print("merged_results:", merged_results)
        # return _predictClass _confidence _predictBbox   --> image
        if merged_results is not None:
            
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
                label_size = (
                draw.textbbox((0, 0), label_text, font=font)[2] - draw.textbbox((0, 0), label_text, font=font)[0], \
                draw.textbbox((0, 0), label_text, font=font)[3] - draw.textbbox((0, 0), label_text, font=font)[1]) \
                    if check_version(PIL.__version__, '9.2.0') else draw.textsize(label_text, font)

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
                
                del draw
            return image_new   
        else:
            return image   

    def detect_slideCvImage(self):
        pass


        
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
            if results[0] is None: 
                return
            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
            top_label, top_conf, top_boxes = self.nms_cross_class(top_label, top_conf, top_boxes, iou_threshold=0.8)
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


