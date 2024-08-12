import random 
import numpy as np 
from PIL import Image
import torch


def cvtColor(image):
    """ Image-file 'gray'  --> 'RGB'

    Args:
        iamge: <'class' Image-file>

    Returns:
        'RGB' image
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        
def resize_image(image, size):
    iw, ih = image.size
    w, h = size
    
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
        
    image = image.resize((nw, nh), image.BICUBIC) 
    new_image = Image.new('RGB', size, (128, 128, 128))   
    new_image.paset(image, ((w-nw)//2, (h-nh)//2))
    
    return new_image, nw, nh


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
def process_input(image):
    image /= 255.0
    return image

def show_condif(**kwargs):
    print('Confihurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('key', 'values'))
    print('-' * 70)
    for key, value in kwargs.item():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
    


def download_weights(backbone, model_dir = "./model_data"):
    import os 
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }

    
        
        