import os, torch, datetime

import torch.utils.data.distributed
from utils.utils import seed_everything, get_classes, download_weights, show_config
from utils.utils_filt import fit_one_epoch
import torch.distributed as dist
from nets.yolo_for_training import YoloBody, Loss, ModelEMA, get_lr_scheduler, set_optimizer_lr
import numpy as np
from utils.callbacks import LossHistory




# load config
from utils.utils import load_cfg
cfg = load_cfg()


if __name__ == '__main__':
    
    # load config
    
    # windows
    os.environ["KMP_DUPLICATE_LIB_OK"] = cfg['train_cfg']['KMP_DUPLICATE_LIB_OK']
    Cuda = cfg['train_cfg']['cuda'] # single gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg['train_cfg']['CUDA_VISIBLE_DEVICES']  # use certain gpu or multi-gpu
    
    seed = cfg['train_cfg']['seed']
    
    
    # -------------- distributed   windows:DP  DDP(default is False)   Ubuntu: DP DDP in Terminal ----------------
    # DP: distributed=False,  Terminal:>>>$ CUDA_VISIBLE_DEVICES=0,1 python train.py
    # DDP: distrubited=True, Termianl:>>>$ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    distributed        = cfg['train_cfg']['distributed'] 
    
    # ------------------  sync_bn only accessible in DDP mode
    sync_bn             = cfg['train_cfg']['sync']
    
    fp16                = cfg['train_cfg']['fp16']
    classes_path        = cfg['dataset_cfg']['classes_path']
    model_path          = cfg['train_cfg']['model_path']
    input_shape         = cfg['train_cfg']['input_shape']
    phi                 = cfg['train_cfg']['phi']
    pretrained          = cfg['train_cfg']['pretrained']  # wether to load the original weights in model_path is None.
    
    mosaic              = cfg['train_cfg']['mosaic']
    mosaic_prob         = cfg['train_cfg']['mosaic_prob']
    mixup               = cfg['train_cfg']['mixup']
    mixup_prob          = cfg['train_cfg']['mixup_prob']
    special_aug_ratio   = cfg['train_cfg']['special_aug_ratio']
    label_smoothing     = cfg['train_cfg']['label_smoothing']
    
    Freeze_train        = cfg['train_cfg']['Freeze_train']
    Freeze_Epoch        = cfg['train_cfg']['Freeze_Epoch']   # fine-tune, freeze model backbone, train model ends layer
    Init_Epoch          = cfg['train_cfg']['Init_Epoch']
    freeze_batch_size   = cfg['train_cfg']['freeze_batch_size']  # batch_size > 2
    
    Unfreeze_Epoch      = cfg['train_cfg']['Unfreeze_Epoch']
    Unfreeze_batch_size = cfg['train_cfg']['Unfreeze_batch_size']
    
    Init_lr             = cfg['train_cfg']['Init_lr']
    Min_lr              = cfg['train_cfg']['Min_lr']
    
    optimizer_type      = cfg['train_cfg']['optimizer_type']
    momentum            = cfg['train_cfg']['momentum']
    weight_decay        = cfg['train_cfg']['weight_decay']
    lr_decay_type       = cfg['train_cfg']['lr_decay_type']
    
    save_period         = cfg['train_cfg']['save_period']
    
    save_dir            = cfg['train_cfg']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    eval_flag           = cfg['train_cfg']['eval_flag']
    eval_period         = cfg['train_cfg']['eval_period']
    
    num_workers         = cfg['train_cfg']['num_workers']
    
    train_annotation_path = cfg['train_cfg']['train_annotation_path']
    val_annotation_path   = cfg['train_cfg']['val_annotation_path']
    
    
    seed_everything(seed)
    
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        # DDP
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("GPU Device Count: ", ngpus_per_node)   
    else:
        device     = torch.cuda('cuda' if torch.cuda.is_available() else 'cpu')   # ?? single gpu, certain gpu, multi-gpus ??
        local_rank = 0
        rank       = 0
    
    class_names, num_classes = get_classes(classes_path)
    
    # whether download the original weights.pth
    if pretrained:
        if distributed:
            if local_rank == 0:
                # single gpu DDP
                download_weights(phi)
            dist.barrier()
        else:
            # DP or single gpu or cpu 
            download_weights(phi)
        
    # model instance
    model = YoloBody(input_shape, num_classes, phi, pretrained)
    
    if model_path != "":
        # load pre_weights.pth
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
            
        model_dict = model.state_dict()  # mdoel state dict need init?? ##1119
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful load key:", str(load_key)[:500], "...'nSuccessful load key name:", len(load_key))
            print("\nFail to load key:", str(no_load_key)[:500], "...\nFail to load key num:", len(no_load_key))
            
    yolo_loss = Loss(model)
    if local_rank    == 0:
        time_str     = datetime.datetime.strFtime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir      = os.path.join(save_dir, 'logs_'+str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None


    if fp16:
        # from torch.cuda.amp import GradScaler
        scaler = torch.amp.grad_scaler()
    else:
        scaler = None
        
    model_train = model.train()    
    # sync bn
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
        
        
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True
            model_train = model_train.cuda()
    
    # weight smooth
    ema = ModelEMA(model_train)

    with open(train_annotation_path, "r", encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, "r", encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    
    if local_rank == 0:
        show_config()
    

    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // Unfreeze_batch_size * Unfreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError("datais not enough ! !")
        wanted_step = wanted_step // (num_train // Unfreeze_batch_size) + 1
        # print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        # print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        # print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    if True:
        Unfreeze_flag = False
        if Freeze_train:
            for param in model.backbone.parameters():
                param.requires_grad = False
  
        batch_size = freeze_batch_size if Freeze_train else Unfreeze_batch_size
        
        # get min-lr max-lr
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == "adam" else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size/nbs * Min_lr, lr_limit_min*1e-2), lr_limit_max*1e-2)
        
        # optimizer
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, torch.nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': torch.optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': torch.optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]        
        

        # lr scheduler
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Unfreeze_Epoch)
        
        # confirm step length
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("data is not enough to train !!")
        
        if ema:
            ema.updates = epoch_step * Init_Epoch  #  why ##1120  
            # ema updates中并没有发生期待的深层非backbone层的权重/梯度更新 ？  ## 1121
        
        
        # datset
        train_dataset = YoloDataset()
        val_dataset = YoloDataset()
        
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
            batch_size    = batch_size // ngpus_per_node
            shuffle       = False 
        else:
            train_sampler = None
            val_sampler   = None
            shuffle       = True 
            
        # dataloader
        gen               = Dataloader(train_dataset, train=True)
        gen_val           = Dataloader(val_dataset, train=False)
            
        # record eval ema
        if local_rank == 0:
            eval_callback = EvalCallback()
        else:
            eval_callback = None
        
        # train
        for epoch in range(Init_Epoch, Unfreeze_Epoch):
            
            # unfreeze train
            if epoch >= Freeze_Epoch and not Unfreeze_flag and Freeze_train:
                
                # get lr by <epoch>
                nbs            = 64
                lr_limit_max   = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min   = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit    = min(max(batch_size/nbs * Init_lr_fit, lr_limit_min), lr_limit_max)
                Min_lr_fit     = min(max(batch_size/nbs * Min_lr, lr_limit_min*1e-2), lr_limit_max*1e-2)
                
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Unfreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True
                
                epoch_step     = num_train // batch_size
                epoch_step_val = num_val // batch_size
                
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("data is not enough to train !!")
                
                if ema:
                    ema.updates = epoch_step * epoch
                if distributed:
                    batch_size  = batch_size // ngpus_per_node
                    
                gen  = Dataloader()
                gen_val = Dataloader()
                
                Unfreeze_flag = True 
                
            
            gen.dataset.epoch_now   = epoch
            gen_val.dataset.epoch_new = epoch
            
            if distributed:
                train_sampler.set_epoch(epoch)
            
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, 
                          epoch_step, epoch_step_val, gen, gen_val, Unfreeze_Epoch, Cuda, fp16, scaler,
                          save_period, save_dir, local_rank)
            
            if distributed:
                dist.barrier()
            
            
            
        if local_rank == 0:
            loss_history.writer.close()
        







