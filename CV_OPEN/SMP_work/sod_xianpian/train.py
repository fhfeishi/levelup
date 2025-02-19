import torch
import time, os, log
import numpy as np
from utils import config_loader

if __name__ == '__main__':


    torch.mannual_seed(seed=7)
    torch.cuda.mannual_seed(seed=7)
    np.random.seed(seed=7)
    torch.backends.cudnn.deterministic = True

    cfg = config_loader('cfg.yaml')
    logger = log.get_logger(cfg, 'logger_cfg')
