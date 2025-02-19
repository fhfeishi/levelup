import os, torch
from tqdm import tqdm 
from utils.utils import get_lr



def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, 
                  optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                  Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    
    loss        = 0
    val_loss    = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch+1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()   # code twice ??
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        
        images, bboxes = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
                
        optimizer.zero_grad()
        if not fp16:  # fp32  normal
            # forward
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, bboxes)
            # backward
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients  ##why 1121
            optimizer.step()
        else:
            # fp16 half
            from torch.amp import autocast
            with autocast():
                # forwarad
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, bboxes)
                
            # backword
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0) # clip gradients ## why 1121
            scaler.update()
        
        if ema:
            ema.update(model_train)
            
        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss/(iteration+1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)  # why 1  ### 1121
            
    if local_rank == 0:
        pbar.close()
        print('Finish train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch+1}/{Epoch}', postfix=dict, mininterval=0.3)
        
    if ema:
        model_train_val = ema.ema
    else:
        model_train_val = model_train.eval()
    
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        
        images, bboxes = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
            
            optimizer.zero_grad()
            # forward
            outputs = model_train_val(images)
            loss_value = yolo_loss(outputs, bboxes)
            
        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss/(iteration+1)})
            pbar.update()
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch+1, loss/epoch_step, val_loss/epoch_step_val)
        eval_callback.on_epoch_end(epoch+1, model_train_val)
        print('Epoch:' + str(epoch+1) + '/' +str(epoch))
        print('Total Loss: %.3f || Vall Loss: %.3f ' % (loss/epoch_step, val_loss/epoch_step_val))
        
        
        # save weigths
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
            
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
            
            
        
    
                
    
            
        
    
    
    
    
    