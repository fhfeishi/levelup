import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 

# cpu inference
def imageSegModel_cpu_output_inference_(input_cvimg, seg_model):
    seg_model.eval()
    with torch.no_grad():
        input_data = np.array(input_cvimg) / 255.0 if not isinstance(input_cvimg, np.ndarray) else input_cvimg/255.0
        input_data = torch.from_numpy(input_data.transpose((2,0,1)).unsqueeze(0))
        output_ = seg_model(input).detach().squeeze().numpy
        # output_ = seg_model(input).detach()[0].numpy
        # value:[0, 1] shape:hwc [:,:ci]---> class[i]_possibility
        return output_


# silent objection detection， classfiy： (_background_,） target
# ## binary_method is ok
# ### image_  mask_   --model-->  output_
def mae_loss(input_, mask_):
    losses = [F.binary_cross_entropy_with_logits(input_[i], mask_) 
              for i in range(len(input_))]
    total_loss = sum(losses)
    return total_loss


# semantic segmentation, classify: _background_, target1, target2, target3, ...
# ## multiClass_method maybe.  should classifiy: (target_nums + 1) _cla_ times.

# condition_a
# ###  dataLoader -> image_, mask_ (ch=1), label_ (ch=cla)  --seg model--> output_
def ce_loss(output_, mask_, cls_weights, num_clas, alpha=.5, gamma=.2):
    n,c,h,w = output_.shape
    nm, hm, wm = mask_.shape  # ch =1
    if h != hm and w != wm:
        output_ = F.interpolate(output_, size=(hm, wm), mode="bilinear", align_corners=True)

    temp_output_ = output_.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = mask_.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_clas)(temp_output_, temp_target)
    return CE_loss
    
def Focal_Loss(output_, mask_, cls_weights, num_cla, alpha=0.5, gamma=2):
    n, c, h, w = output_.size()
    nt, ht, wt = mask_.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_output_ = output_.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_mask_ = mask_.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_cla, reduction='none')(temp_output_, temp_mask_)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def dice_loss(output_, label_, beta=1, smooth=1e-5):
    n,c,h,w = output_.size() # torch.tensor
    nl,hl,wl,cl = label_.size()  # nhwc?  np.ndarray --> hwc  --torch batch--> bhwc 
    if h != hl and w != wl:
        output_ = F.interpolate(output_, size=(hl, wl), mode='bilineat', align_corners=True)
    
    tmp_output_ = torch.softmax(output_.transpose(1,2).transpose(2,3).contiguous().view(n, -1, c), -1)
    tmp_label_ = label_.view(n, -1, cl)
    
    tp = torch.sum(tmp_label_[...,:-1] * tmp_output_, axis=[0,1])
    fp = torch.sum(tmp_output_                       , axis=[0,1]) - tp
    fn = torch.sum(tmp_label_[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss
    
    
# condition_b
# ### dataLoader -> image_, mask_ --seg model--> output_
# label_  = get_label_(mask_, num_cla, )

def build_label_(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)
