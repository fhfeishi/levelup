import os
import numpy as np
import torch
import random 
import PIL.Image as Image
from torch.utils.data.dataset import Dataset

class u2netssDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(u2netssDataset, self).__init__()
        
        self.annotations_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        annotation_line = self.annotations_lines
        name = annotation_line.split()[0]
        
        
        
    





