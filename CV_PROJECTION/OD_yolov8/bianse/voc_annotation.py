import random
import numpy as np  
import xml.etree.ElementTree as ET

from utils.utils import get_classes


annotation_mode = 0

classes_path = r"model_data\my_classes.txt"


trainval_percent = 0.9
train_percent = 0.9

VOCdevkit_path = 'VOCdevkit'
VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
classes, _ = get_classes(classes_path)

photo_nums  = np.zeros(len(VOCdevkit_sets))

nums        = np.zeros(len(classes))

#   dataFileName  ->  ImageSet/Main   train.txt val.txt
