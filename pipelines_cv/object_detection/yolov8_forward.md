# yolov8_s forward  
`label_list['head', 'helmet', 'safetybelt']`
`num_classes = 3`

```python
#  #############   forward   ###########################
def forward(self, x):

    # yolov8 s   : self.deep_mul = 1.0
    #            : self.reg_max  = 16

    #  backbone -> 三个有效特征层    # backbone
    feat1, feat2, feat3 = self.backbone.forward(x)
    # feat1 shape: n, 1024, 20, 20  n: batch_size
    # feat2 shape: n, 512,  40, 40
    # feat3 shape: n, 256,  80, 80

    # 加强特征提取网络FPN                        nchw  
    P5_upsample = self.upsample(feat3)      # n, 1024, 20, 20 => n, 1024, 40, 40
    P4 = torch.cat([P5_upsample, feat2], 1) # n, 1024+512, 40, 40
    P4 = self.conv3_for_upsample1(P4)       # => n, 512, 40, 40

    P4_upsample = self.upsample(P4)         # n, 512, 40, 40 => n, 512, 80, 80
    P3 = torch.cat([P4_upsample, feat1], 1) # => n, 512+256, 80, 80
    P3 = self.conv3_for_upsample2(P3)       # => n, 256, 80, 80

    P3_downsample = self.down_sample1(P3)   # => n, 256, 40, 40
    P4 = torch.cat([P3_downsample, P4], 1)  # => n, 512+256, 40, 40
    P4 = self.conv3_for_downsample1(P4)     # => n, 512, 40, 40

    P4_downsample = self.down_sample2(P4)    # => n, 512, 20, 20
    P5 = torch.cat([P4_downsample, feat3], 1)# => n, 1024+512, 20, 20
    P5 = self.conv3_for_downsample2(P5)      # => n, 1024, 20, 20
    
    # 加强特征网络FPN的输出：
    # P3 shape: n, 256,  80, 80
    # P4 shape: n, 512,  40, 40
    # P5 shape: n, 1024, 20, 20

    # Head
    shape = P3.shape  # n, 256, 80, 80
    x = [P3, P4, P5]
    for i in range(self.nl): # self.nl = 3
        # P3 => n, 3+16*4, 80, 80     num_classes + self.reg_max * 4, 80, 80
        # P4 => n, 3+16*4, 40, 40     num_classes + self.reg_max * 4, 40, 40
        # P5 => n, 3+16*4, 20, 20     num_classes + self.reg_max * 4, 20, 20
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) # 仅改变通道数的卷积操作，然后cat
        # self.cv2 -> 输出通道数是 64， 负责边界框回归,  
        # self.cv3 -> 输出通道数为 3，  负责类别预测

    
    if self.shape != shape:  # self.shape 初始值是None
        self.anchors, self.strides = (x.transpose(0, 1) for x in 
                                        make_anchors(x, self.stride, 0.5))
        # make-anchor就是将宽度W高度H的特征图均分成块，
        # 比如将 n,c,20,20 的输入特征图分成20*20块，然后取块的中心点(off_set=0.5)作为锚点, 20*20=400个锚点
        # 每个锚点anchors有一个步长strides，对应到输入特征图的图像坐标系，用于位置对齐
        #      xy (2,8400)  (1,8400)         <---    (8400,2)  (8400,1)
        
        self.shape = shape   # n, 256, 80, 80

    # make anchor
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        # feats: tensor([n,c,80,80],[n,c,40,40],[n,c,20,20])
        # strides: [8, 16, 32]   # 640-> 80 40 20 -> 256 /8  16 32
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij') 
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)
        #        (8400,2)    (8400,1)   8,16,32

    # 80*80+60*60+20*20 = 8400
    # torch.cat()          =>  n, 3+64, 8400       | num_classes + self.reg_max * 4 , 8400 
    # torch.Tensor.split() =>  
    # cls n,num_classes, 8400;
    # box n,self.reg_max * 4, 8400
    box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
        (self.reg_max * 4, self.num_classes), 1)

    # 将边界框预测通过DFL模块处理，将[batch_size, 64, 8400]的输出转换为[batch_size, 4, 8400]
    # DFL通过将每个坐标(x,y,w,h)建模为分布并取加权和，提高边界框回归精度
    dbox = self.dfl(box)

    # DFL  -->  for  reg loss  或者说本身这些值就是[0,1]之间的数值
    class DFL(nn.Module):    
        def __init__(self, c1=16):
            super().__init__()
            self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
            x = torch.arange(c1, dtype=torch.float) # x = torch.arange(16)
            self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
            #                          [[[w0], [w1], [w2], ...., [w15]]]
            #                          [[[0], [1], [2], ...., [15]]]  
            #                           self.conv == dot_product
            #                             [n, 64, 8400]  
            self.c1 = c1
        def forward(self, x):          # self.reg_max = 16
            n, c, a = x.shape          # n, 64, 8400               # 用16个值
            t1 = x.view(n,4,self.c1,a) # n, 4, 16, 8400
            t2 = t1.transpose(2, 1)    # n, 16, 4, 8400
            t3 = t2.softmax(1)         # n, 16, 4, 8400  
            # softmax(1)(n个16通道4x8400的数组)
            # => n*16个4*8400的数组，
            # => 沿通道维度(0123...15)方向上crop一个列向量进行softmax
            # => n * (4x8400) 个这样的列向量（序列、数列）  
            t4 = self.conv(t3)         # n, 1, 4, 8400
            t5 = t4.view(n, 4, a)      # n, 4, 8400                 # 预测4个值
            return t5
         (n,4,8400)ltrb (n,ncls,8400) outs (2,8400) x,y    (1,8400) 8 16 32
    return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)


#  #############   training   ###########################
# 1、计算loss所需内容
#   计算loss实际上是网络的预测结果和网络的真实结果的对比。
#   和网络的预测结果一样，网络的损失也由两个部分组成，分别是回归部分、种类部分。回归部分是特征点的回归参数判断、种类部分是特征点包含的物体的种类。
# 2、正样本的匹配过程
#   在YoloV8中，训练时正样本的匹配过程可以分为三部分。
#       - 根据空间距离判断特征点是否在真实框中。
#       - 根据代价函数判断特征点是否在真实框内的topk中。
#       - 去重等后处理。
#   所谓正样本匹配，就是寻找哪些特征点被认为有对应的真实框，并且负责这个真实框的预测。
#   a、判断特征点是否在预测框内
# 根据空间距离判断特征带你是否在预测框中，yolov8会对每个真实框进行粗匹配，找到哪些特征点上的哪些先验框可以负责该真实框的预测
# 代码根据真实框与特征点的坐标情况，利用特征点坐标减去真实框左上角，利用真实框右下角减去特征点坐标，如果这几个值都大于0则特征点在真实框内






















#  #############   loss_fn   ###########################
# Criterion class for computing training losses
class Loss:
    def __init__(self, model):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = model.stride  # model strides
        self.nc = model.num_classes  # number of classes
        self.no = model.no
        self.reg_max = model.reg_max

        self.use_dfl = model.reg_max > 1
        roll_out_thr = 64

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss = BboxLoss(model.reg_max - 1, use_dfl=self.use_dfl)
        self.proj = torch.arange(model.reg_max, dtype=torch.float)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=targets.device)
        else:
            # 获得图像索引
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
            # 对batch进行循环，然后赋值
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            # 缩放到原图大小。
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            # batch, anchors, channels
            b, a, c = pred_dist.shape
            # DFL的解码
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.to(pred_dist.device).type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        # 然后解码获得预测框
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        # 获得使用的device
        device = preds[1].device
        # box, cls, dfl三部分的损失
        loss = torch.zeros(3, device=device)
        # 获得特征，并进行划分
        feats = preds[2] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # bs, num_classes + self.reg_max * 4 , 8400 =>  cls bs, num_classes, 8400;
        #                                               box bs, self.reg_max * 4, 8400
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # 获得batch size与dtype
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # 获得输入图片大小
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]
        # 获得anchors点和步长对应的tensor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 把一个batch中的东西弄一个矩阵
        # 0为属于第几个图片
        # 1为种类
        # 2:为框的坐标
        targets = torch.cat((batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1), batch[:, 2:]), 1)
        # 先进行初步的处理，对输入进来的gt进行padding，到最大数量，并把框的坐标进行缩放
        # bs, max_boxes_num, 5
        targets = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # bs, max_boxes_num, 5 => bs, max_boxes_num, 1 ; bs, max_boxes_num, 4
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # 求哪些框是有目标的，哪些是填充的
        # bs, max_boxes_num
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        # 对预测结果进行解码，获得预测框
        # bs, 8400, 4
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # 对预测框与真实框进行分配
        # target_bboxes     bs, 8400, 4
        # target_scores     bs, 8400, 80
        # fg_mask           bs, 8400
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # 计算分类的损失
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # 计算bbox的损失
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        return loss.sum()  # loss(box, cls, dfl) # * batch_size



```
































