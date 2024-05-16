import torch
import numpy as np
import os
from transformers import AdamW
import random
from evaluate import load
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import math
"""
multi GPU then
if device.type == 'cuda' and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
"""

class InitProcessor:
    def __init__(self, seed = 42):
        seed_val = seed
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        
    def get_device(self, config, use_gpu = True, CUDA_VISIBLE_DEVICES = "0, 1, 2"):
        if use_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
            device = torch.device("cuda:0")
            config.device = "cuda:0"
            return config
   
        device = torch.device("cpu")
        config.device = "cpu"
        return config
    
    def get_optimizer(self, optimizer_name, model, lr, eps = 1e-8, alpha = 1):
        optimizer_1 = None
        optimizer_2 = None
        param_1 = [
            {
                "params": [p for n, p in model.named_parameters() if "query" in n or "hidden" in n]
            }
        ]
        param_2 = [
            {
                "params": [p for n, p in model.named_parameters() if "attention" in n]
            }
        ]
        if optimizer_name == "AdamW":
            optimizer_1 = AdamW(param_1,
                                lr = lr * alpha,
                                eps = eps)
            optimizer_2 = AdamW(param_2,
                                lr = lr,
                                eps = eps)
        if optimizer_1 != None and optimizer_2 != None:
            return optimizer_1, optimizer_2
        else:
            print("None optimizer name")
            
    def _get_custom_schedule_lr_lambda(self, current_step: int, *, num_warmup_steps: int, timescale: int = None):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps * 3:
            return 1
        else:
            num_warmup_steps *= 3
        shift = timescale - num_warmup_steps
        decay = 1.0 / math.sqrt((current_step + shift) / timescale)
        return decay
        
    def get_custom_scheduler(self, optimizer, num_warmup_steps = 0, timescale = None, last_epoch = -1):
        if timescale is None:
            timescale = num_warmup_steps
        lr_lambda = partial(self._get_custom_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
        return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)