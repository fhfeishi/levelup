# 尽量避免了padding
# 前置条件就是：目标-bbox  < crop-size， 不然就得resize处理
# 从标注文件的bbox找到一个point，然后根据这个point还原到crop-size， 然后裁剪
# 或者直接根据目标的bbox随机生成一个crop-size-box， 然后再裁剪，更新坐标，更新文件名
#    问题就是说如果有多个目标怎么办，会存在 裁剪 bbox的可能

import random 
import xml.etree.ElementTree as ET
import numpy as np

# 避免了裁剪bbox
class crop_ddataset():
    def __init__(self, image_path, xml_path, seed=42, window=640):
        self.image_path = image_path
        self.xml_path = xml_path
        self.window = window
        # 随机种子
        random.seed(seed)
        # parse xml file
        self.bboxes = self.parse_xml()

        # single-box or  box-group
        crop_mode = self.get_mode()
    
    def parse_xml(self):
        # 解析XML文件
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        # 获取所有bbox的坐标
        bboxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bboxes.append((xmin, ymin, xmax, ymax))
                
    def get_mode(self):
        if len(self.bboxes) == 1:
            return "mode-single"
        elif len(self.bboxes) == 2:
            w1, h1 = get_box_wh(self.bboxes[0])
            w2, h2 = get_box_wh(self.bboxes[1])
            enclosing_w, enclosing_h, _ = get_enclosing_bbox(self.bboxes)
            if enclosing_w>(640+w1+w2) or enclosing_h>(640+h1+h2):
                return "mode-single"
            else:
                return "mode-groups"

        elif len(self.bboxes) > 2: # 看怎么处理一下  好麻烦啊   
            bbox_num = len(self.bboxes)
            # 判断bbox是否相邻
            enclosing_w, enclosing_h, _ = get_enclosing_bbox(self.bboxes)
            if enclosing_w>640 and enclosing_h>640:
                return "mode-single"
            else:
                return "mode-groups"
        else:
            print("no bbox!")
            
    def group_boxes(self):
        single_boxes = []
        group_boxes = []

        if len(self.bboxes) > 2:
            # Sort boxes by area to prioritize smaller boxes fitting together
            sorted_boxes = sorted(self.bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        while sorted_boxes:
            current_box = sorted_boxes.pop(0)
            potential_group = [current_box]
            potential_area = self.window * self.window
            
            for other_box in sorted(sorted_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])):
                if self.can_fit_together(potential_group + [other_box]):
                    potential_group.append(other_box)

            if len(potential_group) == 1:
                single_boxes.append({"mode": "single", "bbox": potential_group})
            else:
                group_boxes.append({"mode": "group", "bbox": potential_group})
                for b in potential_group:
                    sorted_boxes.remove(b)  # Remove grouped boxes from consideration

        return single_boxes, group_boxes
    
    # 处理 mode-single
    def get_cbox(self, bbox):
        # 解析bbox信息
        xmin, ymin, xmax, ymax = bbox
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        # 计算窗口的最大可能左上角坐标
        max_win_xmin = xmin - (self.window - bbox_width)
        max_win_ymin = ymin - (self.window - bbox_height)
        
        # 确保最大可能坐标不小于0
        max_win_xmin = max(0, max_win_xmin)
        max_win_ymin = max(0, max_win_ymin)
        
        # 在最大可能范围内随机选择窗口的左上角坐标
        win_xmin = random.randint(max_win_xmin, xmin)
        win_ymin = random.randint(max_win_ymin, ymin)
        
        # 计算窗口的右下角坐标
        win_xmax = win_xmin + self.window
        win_ymax = win_ymin + self.window
    
        return (win_xmin, win_ymin, win_xmax, win_ymax)
    
    def get_cboxes(self):
        pass
    def cut_ddataset(self):
        # cut jpg xml
        pass

# 这里如果是某一类目标很关键，可以尽可能保证 在一个threshold范围内的 得到不裁剪zip(crop_image, crop_mask)
# semantic segmentation

# Dataset
#    __getitem__
#    get_random_data()
#    get_fixedWin_data()



# object detection


# 没有鲁棒性，不够好，所以还是  box[640, 640]  all/70% : in-size-bbox[i] > 7% bbox[i]  实现一下 五一。。
@staticmethod
def get_enclosing_bbox(bboxes):
    # 计算所有bbox的最小外接矩形
    xmin = min(bbox[0] for bbox in bboxes)
    ymin = min(bbox[1] for bbox in bboxes)
    xmax = max(bbox[2] for bbox in bboxes)
    ymax = max(bbox[3] for bbox in bboxes)
    enclosing_bbox = [xmin, ymin, xmax, ymax]
    enclosing_w, enclosing_h = abs(xmin-xmax), abs(ymin-ymax)
    return enclosing_w, enclosing_h, enclosing_bbox

@staticmethod
def get_box_wh(box):
    [xmin, ymin, xmax, ymax] = box
    box_w, box_h = abs(xmin-xmax), abs(ymin-ymax)
    return box_w, box_h 


# 输入一个bbox，  返回一个cbox
# 适用于两种情形， xml文件中只有一个bbox， xml中的bbox间隔足够远
def get_cbox_random_a(bbox, img_size=(640, 640), seed=42):
    
    random.seed(seed)  # 设置随机数种子
    
    # 解析bbox信息
    xmin, ymin, xmax, ymax = bbox
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    
    # 计算窗口的最大可能左上角坐标
    max_win_xmin = xmin - (img_size[0] - bbox_width)
    max_win_ymin = ymin - (img_size[1] - bbox_height)
    
    # 确保最大可能坐标不小于0
    max_win_xmin = max(0, max_win_xmin)
    max_win_ymin = max(0, max_win_ymin)
    
    # 在最大可能范围内随机选择窗口的左上角坐标
    win_xmin = random.randint(max_win_xmin, xmin)
    win_ymin = random.randint(max_win_ymin, ymin)
    
    # 计算窗口的右下角坐标
    win_xmax = win_xmin + img_size[0]
    win_ymax = win_ymin + img_size[1]
    
    return (win_xmin, win_ymin, win_xmax, win_ymax)

# 复杂的情形
# xml中有多个bbox， 间隔近

def get_bbox_windows(xml_file, img_size=(640, 640)):
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    distance_threshold = img_size[0]

    # 获取所有bbox的坐标
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
    
    # 初始化窗口列表
    windows = []
    
    # 遍历所有bbox，为每个bbox生成一个窗口
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        center_x = xmin + bbox_width // 2
        center_y = ymin + bbox_height // 2
        
        # 计算窗口的左上角和右下角坐标
        win_xmin = max(0, center_x - img_size[0] // 2)
        win_ymin = max(0, center_y - img_size[1] // 2)
        win_xmax = win_xmin + img_size[0]
        win_ymax = win_ymin + img_size[1]
        
        # 检查窗口是否与其他bbox相交
        intersects = False
        for other_bbox in bboxes:
            if other_bbox != bbox:
                other_xmin, other_ymin, other_xmax, other_ymax = other_bbox
                if not (win_xmax <= other_xmin or win_xmin >= other_xmax or win_ymax <= other_ymin or win_ymin >= other_ymax):
                    intersects = True
                    break
        
        # 如果窗口与其他bbox相交，返回提示
        if intersects:
            return "此文件需要重新处理"
        
        # 否则，添加窗口到列表
        windows.append((win_xmin, win_ymin, win_xmax, win_ymax))
    
    return windows
  

if __name__ == '__main__':
    

    print("done")



