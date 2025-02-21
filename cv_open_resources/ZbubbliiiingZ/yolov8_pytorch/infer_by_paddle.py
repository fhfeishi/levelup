import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import paddle.inference as paddle_infer
import paddle, os

# load rgb-image to model-input-data
def load_image(img_path, rsize_shape=(640,640)):
    origin_img = Image.open(img_path).convert("RGB")
    origin_size = origin_img.size
    resized_img = origin_img.resize(rsize_shape) 
    im_data = np.array(resized_img, dtype=np.float32) / 255.0
    im_data = im_data.transpose(2,0,1)
    im_data = np.expand_dims(im_data, axis=0)
    return resized_img, im_data, origin_img, origin_size


def nms(boxes,scores, iou_threshold=0.45):
    boxes = paddle.to_tensor(boxes)
    scores = paddle.to_tensor(scores)
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    area = (x2 - x1) * (y2 - y1)
    order_ = paddle.argsort(scores, descending=True)
    
    keep = []
    while order_.shape[0] > 0:
        i = order_[0].item()
        keep.append(i)
    
    

