import math 
import torch

# silient-objection-objection   background 0, target 1

class criterion_sod_module(torch.nn.Module):
    """Loss calculation strategy for U2Net salient object detection.

    Args:
        preds (Tensor): Predictions from the U2Net.
        target (Tensor): Ground truth labels.
        mode (str): Specifies the combination of loss functions to use. Options are:
            'bce_loss'
            'bce_loss + dice_loss'
            'bce_loss + focus_loss'
            'bce_loss + dice_loss + focus_loss'
    """
    def __init__(self, mode = 'bce_loss'):
        super(credits, self).__init__()
        self.mode = mode
        
    def forward(self, inputs, target):
        bce_losses = [torch.nn.functional.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
        total_bce_loss = sum(bce_losses)

        if 'dice_loss' in self.mode:
            dice_losses = []
            smooth = 1e-5
            for i in range(len(inputs)):
                preds_sigmoid = torch.sigmoid(inputs[i])
                intersection = torch.sum(preds_sigmoid * target)
                dice_loss = 1 - (2. * intersection + smooth) / (torch.sum(preds_sigmoid) + torch.sum(target) + smooth)
                dice_losses.append(dice_loss)
            total_dice_loss = sum(dice_losses)

        if 'focus_loss' in self.mode:
            focus_losses = []
            beta = 2  # Parameter beta for adjusting focus loss intensity
            for i in range(len(inputs)):
                preds_sigmoid = torch.sigmoid(inputs[i])
                focus_loss = torch.pow(torch.abs(target - preds_sigmoid), beta) * F.binary_cross_entropy_with_logits(inputs[i], target, reduction='none')
                focus_losses.append(focus_loss.mean())
            total_focus_loss = sum(focus_losses)

        loss = total_bce_loss
        if 'dice_loss' in self.mode:
            loss += total_dice_loss
        if 'focus_loss' in self.mode:
            loss += total_focus_loss

        return loss
   

   
def criterion_u2net_sod(inputs, target, mode='bce_loss', show_loss_mode=False):
    losses = [torch.nn.functional.binary_cross_entropy_with_logits
              (inputs[i], target) for i in range(len(inputs))]
    bce_loss = losses[0] * 1.5 + sum(losses[1:])
    total_loss = bce_loss
    if 'dice_loss' in mode:
        dice_losses = []
        smooth = 1e-5
        for i in range(len(inputs)):
            preds_sigmoid = torch.sigmoid(inputs[i])
            intersection = torch.sum(preds_sigmoid * target)
            dice_loss = 1 - (2. * intersection + smooth) / (torch.sum(preds_sigmoid) + torch.sum(target) + smooth)
            dice_losses.append(dice_loss)
        total_dice_loss = sum(dice_losses)
        total_loss += total_dice_loss   
    elif 'focus_loss' in mode:
        focus_losses = []
        beta = 2  # Parameter beta for adjusting focus loss intensity
        for i in range(len(inputs)):
            preds_sigmoid = torch.sigmoid(inputs[i])
            focus_loss = torch.pow(torch.abs(target - preds_sigmoid), beta) * F.binary_cross_entropy_with_logits(inputs[i], target, reduction='none')
            focus_losses.append(focus_loss.mean())
        total_focus_loss = sum(focus_losses)
        total_loss += total_focus_loss
    else:
        print("mode = 'bce_loss' + None or 'dice_loss' or 'focus_loss', but get f'{mode}'")
    if show_loss_mode is True:
        print(mode)
    
    return total_loss
    
 
 
      
def dice_loss(pred, target, smooth=1):
    # sigmoid
    pred = torch.sigmoid(pred)

    # flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # 
    intersection = (pred_flat * target_flat).sum()
    union = (pred_flat + target_flat).sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # dice loss
    
# for u2net  #status#:ok
def bce_loss(inputs, target):
    losses = [torch.nn.functional.binary_cross_entropy_with_logits
              (inputs[i], target) for i in range(len(inputs))]
    bce_loss = sum(losses)
    
    return bce_loss





