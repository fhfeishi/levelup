import torch
import torch.nn as nn


# build model

def C2f():
    pass


class Backbone(nn.Module):
    pass 



class DFL(torch.nn.Module):
    # DFL module
    # Distribution Focal Loss (DFL) proposed in Generalized
    # Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16): 
        super().__init__()
        self.conv = torch.nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x         = torch.arange(c1, dtype=torch.float32)
        self.conv.weight.data[:] = torch.nn.Parameter(x.view(1, c1, 1, 1))
        self.c1   = c1
        
    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # softmax -> (0, 16) percent  -> int
        return self.conv(x.view(b, 4, self.c1, a).transpose(2,1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
        

class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict             = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict             = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        deep_width_dict        = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
        
        base_channels          = int(wid_mul * 64)  # 64
        base_depth             = max(round(deep_mul * 3))  # 3
        # input: 3, 640, 640
        # backbone  -> feat1 256,80,80   
        #              feat2 512,40,40 
        #              feat3 1024*deep_mul,20,20
        self.backbone          = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
        
        # feature aug extract net
        self.upsample          = nn.Upsample(scale_factor=2, mode="nearest")
        
        # 1024*deep_mul + 512,40,40  -> 512,40,40
        self.conv3_for_upsample1 = C2f(int(base_channels*16*deep_mul) + base_channels*8, 
                                      base_channels*8, base_depth, shortcut=False)
        # 768,80,80  -> 256,80,80
        self.conv3_for_upsample2 = C2f(base_channels*8 + base_channels*4, base_channels*4,
                                       base_depth, shortcut=False)
        
        # 256,80,80 -> 256,40,40
        self.down_sample1        = Conv(base_channels*4, base_channels*4, 3, 2)
        
    
    def forward(self, x):
        
        # backbone 
        feat1, feat2, feat3         = self.backbone.forward(x)
        
        # feature aug extract net
        # 1024*deep_mul,20,20 -> 1024*deep_mul,40,40
        P5_upsample                 = self.upsample(feat3)
        # 1024*deep_mul,40,40 cat 512,40,40  -> 1024*deep_mul+512,40,40
        P4                          = torch.cat([P5_upsample, feat2], 1)
        # 1024*deep_mul+512,40,40 -> 512,40,40
        P4                          = self.conv3_for_upsample1(P4)
        
        # 512,40,40 -> 512,80,80
        P4_upsample                 = self.upsample(P4)
        # 512,80,80 cat 256,80,80 -> 768,80,80
        P3                          = torch.cat([P4_upsample, feat1], 1) 
        # 768,80,80 -> 256,80,80
        P3                          = self.conv3_for_upsample2(P3)
        
        # 256,80,80 -> 256,40,40
        P3_downsample               = self.down_sample1(P3)
        # 512,40,40 cat 256,40,40 -> 768,40,40
        P4                          = torch.cat([P3_downsample, P4], 1)
        # 768,40,40  ->  512,20,20
        P4                          = self.conv3_for_downsample1(P4)
        
        # 512,40,40 -> 512,20,20
        P4_downsample               = self.down_sample2(P4)
        # 512,20,20 cat 1024*deep_mul,20,20  -> 1024*deep_mul+512,20,20
        P5                          = torch.cat([P4_downsample, feat3], 1)
        # 1024*deep_mul + 512, 20,20 -> 1024*deep_mul,20,20
        P5                          = self.conv3_for_downsample2(P5)
        
        # aug feature extract
        # P3 256,80,80
        # P2 512,40,40
        # P5 1024*deep_mul,20,20
        shape                       = P3.shape # BCHW
        
        # P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400; 
        #                                           box self.reg_max * 4, 8400
        box, cls        = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox            = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)
        
        
        
        
        
        

