import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    # kernel padding dilation
    # pad input features, Same
    if d > 1 :
        # actual kernel_pad
        k = d*(k-1)+1 if isinstance(k, int) else [d * (x-1) + 1 for x in k]  # why use [ for k]   ## 1119
    if p is None:
        # auto-pad 
        p = k // 2 if isinstance(k, int)  else [x //2 for x in k]
    return p

class SiLU(nn.Module):
    @staticmethod    # why use staticmethod, what use  ## 1119
    def forward(x):
        return x * torch.sigmoid(x)
    
    
        
