import random, torch, cv2
import numpy as np


import yaml
def load_cfg():
    with open('config.yaml', "r", encoding='utf-8') as f:
        return yaml.safe_load(f)
    

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def get_classes(cls_path):
    with open(cls_path, "r", encoding='utf-8') as f:
        class_names = f.readline()
    classes_names = [c.strip() for c in class_names]
    return classes_names, len(classes_names)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def download_weights(phi, model_dir = "./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
        "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
        "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
        "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
        "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth', 
    }
    url = download_urls[phi]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    load_state_dict_from_url(url, model_dir)


def show_config():
    pass 
