import torch, math 
import torch.nn as nn
import numpy as np
from functools import partial
from copy import deepcopy



class YoloBody(nn.Module):
    
    pass 



# criterion class for computing training losses
class Loss(object):
    def __init__(self, model):
        self.bce       = nn.BCEWithLogitsLoss(reduction='none')
        self.tride     = model.stride # model strides
        self.nc        = model.num_classes # model classes num
        self.no        = model.no
        self.reg_max   = model.reg_max
         
        self.use_dfl   = model.reg_max > 1
        roll_out_thr   = 64
        
        self.assigner  = TaskAlignAssigner(tpk=10,
                                          num_classes=self.nc,
                                          alpha=0.5,
                                          beta=6.0,
                                          roll_out_thr=roll_out_thr)
        self.bbox_loss = BboxLoss(model.reg_max -1 , use_dfl=self.use_dfl)
        self.proj      = torch.arange(model.reg_max, dtype=torch.float)
        
    def processes(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out        = torch.zeros(batch_size, 0, 5, device=targets.device)
        else:
            i          = targets[:0]
            _, counts  = i.unique(return_counts=True)
            out        = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
            for j in range(batch_size):
                matches = i==j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
            return out
        
    def bbox_decode(self, anchor_points, pred_dist):
        
        if self.use_dfl:
            # batch_size, anchors, channels
            b, a, c     = pred_dist.shape
            # DFL decode
            pred_dist   = pred_dist.view(b, a, c//4).softmax(3).matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
            # pred_dist   = pred_dist.view(b, a, c//4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist   = (pred_dist.view(b, a, c//4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1 , 1, -1, 1)).sum(2)
        # get pred_bbox
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    
    def __calll__(self, preds, batch):
        # get device
        device  = preds[1].device
        # box-loss, cls-loss, dfl-loss
        
        pass  
        
        
        
        
        
def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA(object):
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """
    
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # create ema
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parametes()).device != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x/tau)) # decay exponential ramp (to help early epoches)
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    def update(self, model):
        # update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            
            msd = de_parallel(model).state_dict()  # model state_dict
            for k,v in self.ema.state_dict().items():
                v *= d
                v += (1-d)*msd[k].detach()
                
    def update_attr(self, model, include=(), exclude=('process_group, reducer')):
        # update EMA attributes
        copy_attr(self.ema, model, include, exclude)
    
    
    

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.005,
                     warmup_lr_ratio=0.1, no_aug_iter_raio=0.05, step_num=10):
    
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters/float(warmup_lr_ratio), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5*(lr-min_lr) * (
                1.0 + math.cos(
                    math.pi
                    * (iters-warmup_total_iters)
                    / (total_iters-warmup_total_iters-no_aug_iter)))
        return lr
    
    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError('step size must above 1')
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr
    
    if lr_decay_type == 'cos':
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start    = max(warmup_lr_ratio*lr, 1e-6)
        no_aug_iter        = min(max(no_aug_iter_raio*total_iters, 1), 15)
        func               = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate         = (min_lr/lr) ** (1/(step_num-1))
        step_size          = total_iters / step_num
        func               = partial(step_lr, lr, decay_rate, step_size)
    
    return func
    

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 


def weight_init():
    pass 



