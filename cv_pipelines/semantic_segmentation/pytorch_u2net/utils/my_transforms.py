import random 
import numpy as np
import torch
import PIL.Image as Image


#---transforms for check input_data--also use in infer---
# ensure input image mode: RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        # h, w, 3
        return image
    else:
        image = image.convert('RGB')
        return image
    
# normalization
def preprocess_input(image):
    image /= 255.0
    return image



# ---transforms for model.train()---------------------
def train_transforms():
    pass







# ---transforms for model.eval()----------------------
def eval_transforms():
    pass












