import scipy.signal
import os, torch, scipy
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt



class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_dir  = log_dir
        
        self.losses   = []
        self.val_loss = []
        
        # Tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # try:
        #     dummpy_input = torch.randn(2,3,input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummpy_input)
        # except:
        #     pass
        
        
    def append_loss(self, epoch, loss, val_loss):
        
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a', encoding='utf-8') as f:
            f.write(str(loss))
            f.write("\n")
            
        with open(os.path.join(self.log_dir, "epoch_val_loss"), "a", encoding='utf-8') as f:
            f.write(str(val_loss))
            f.write('\n')
            
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()
        
    def loss_plot(self):
        iters = range(len(self.losses))
        
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        
        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2 ,label='smooth train loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
        # except:
        #     pass
        
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='uper right')
        
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        
        plt.cla()
        plt.close('all')