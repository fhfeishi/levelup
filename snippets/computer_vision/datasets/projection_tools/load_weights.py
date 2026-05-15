
# model-train-mode:   cpu   single-gpu              dataparallel-gpus           distribute-dataparallel  ?
#        model_best.pth                      model_best.pth ('module.' + ...)
"""
# save
save_file = {"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args}

torch.save(save_file, f"save_weights/model_{epoch}.pth")

"""


# model-inferance  or  rerun-from-checkpoint    load-checkpoint
# mode: cpu single-gpu
"""
checkpoint = torch.load(args.resume, map_location='cpu')
new_state_dict = OrderedDict()
state_dict = checkpoint['model']
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)

optimizer.load_state_dict(checkpoint['optimizer'])
lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
args.start_epoch = checkpoint['epoch'] + 1

"""




# # mode: dataparallel-gpus
"""  
checkpoint = torch.load(args.resume, map_location='cpu')
new_state_dict = OrderedDict()
state_dict = checkpoint['model']
for k, v in state_dict.items():
    if not k.startswith('module.'):
        new_state_dict['module.' + k] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)

optimizer.load_state_dict(checkpoint['optimizer'])
lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
args.start_epoch = checkpoint['epoch'] + 1

""" 

