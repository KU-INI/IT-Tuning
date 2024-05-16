from transformers import get_scheduler, get_constant_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import torch
import torch.nn as nn
import random
import numpy as np
import argparse

from utils import Data, record_training, InitProcessor, validation


parser = argparse.ArgumentParser(description='train')

parser.add_argument('--input_dir', type = str, default = None, required=True)
parser.add_argument('--output_dir', type = str, default = './')
parser.add_argument('--save_dir', type = str, default = './')
#save epoch
#3 means save model each 3 epoch
parser.add_argument('--save', '-s', type = int, default = -1)
parser.add_argument('--epochs', '-e', type = int, default = 100)
parser.add_argument('--batch_size', '-b', type = int, default = 16)
parser.add_argument('--lr_rate', '-lr', type = float, default = 2e-5)
parser.add_argument('--use_gpu', '-g', default = False, action = 'store_true')
parser.add_argument('--task_name', type = str, required = True)
parser.add_argument('--model_type', default = "gpt2-medium")
parser.add_argument('--seq_len', type = int)
parser.add_argument('--rank', type = int, default = 1)
parser.add_argument('--alpha', '-a', type = float, default = 1)
parser.add_argument('--scheduler', '-sch', type = str, default = "linear")

args = parser.parse_args()
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
SAVE = args.save
SAVE_DIR = args.save_dir
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR_RATE = args.lr_rate
GPU = args.use_gpu
MODEL_TYPE = args.model_type
TASK_NAME = args.task_name
RANK = args.rank
SEQ_LEN = args.seq_len
ALPHA = args.alpha
SCHEDULER = args.scheduler

if "gpt2" in MODEL_TYPE:
    from ittune_gpt2 import ittuneGPT2LMHeadModell as Model
    from transformers import GPT2Config as Config
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_TYPE, padding_side = "left")
    
def run():
    init = InitProcessor(42)
    ###################################data###################################
    global BATCH_SIZE
    DATA_NAME = SCHEDULER  + "_b:" + str(BATCH_SIZE) + "_lr:" + str(LR_RATE) + "_e:" +  str(EPOCHS) + "_a:" + str(ALPHA) + "_r:" + str(RANK)
    
    origin_batch = BATCH_SIZE
    if BATCH_SIZE > 32:
        BATCH_SIZE = 32
    #validation dataset has fixed batch size 1.
    train_dataloader, valid_dataloader = Data(TASK_NAME, INPUT_DIR,  BATCH_SIZE)
    
    config = Config().from_pretrained(MODEL_TYPE)
    config = init.get_device(config)
    config.seq_len = SEQ_LEN
    config.rank = RANK
    config.pad_token_id = tokenizer.eos_token_id
    model = Model.from_pretrained(MODEL_TYPE,
                                  config = config)
    ###################################model###################################
    model.to(config.device)
    
    trainable_param = ["vector_"]

    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in trainable_param:
            if i in name:
                param.requires_grad = True
    
    optimizer_1, optimizer_2 = init.get_optimizer(optimizer_name = "AdamW", model = model, lr = LR_RATE, alpha = ALPHA)
    total_steps = len(train_dataloader) * EPOCHS
    
    if SCHEDULER == "linear":
        scheduler_1 = get_scheduler(name = 'linear',
                              optimizer = optimizer_1, 
                              num_warmup_steps = int(total_steps * 0.06),
                              num_training_steps = total_steps)
        scheduler_2 = get_scheduler(name = 'linear',
                              optimizer = optimizer_2, 
                              num_warmup_steps = int(total_steps * 0.06),
                              num_training_steps = total_steps)
    elif SCHEDULER == "constant":
        scheduler_1 = get_constant_schedule_with_warmup(
                              optimizer = optimizer_1, 
                              num_warmup_steps = int(total_steps * 0.06))
        scheduler_2 = get_constant_schedule_with_warmup(
                              optimizer = optimizer_2, 
                              num_warmup_steps = int(total_steps * 0.06))
    elif SCHEDULER == "cosine_restarts":
        scheduler_1 = get_cosine_with_hard_restarts_schedule_with_warmup(
                              optimizer = optimizer_1, 
                              num_warmup_steps = int(total_steps * 0.06),
                              num_training_steps = total_steps,
                              num_cycles = int(EPOCHS / 4))
        scheduler_2 = get_cosine_with_hard_restarts_schedule_with_warmup(
                              optimizer = optimizer_2, 
                              num_warmup_steps = int(total_steps * 0.06),
                              num_training_steps = total_steps,
                              num_cycles = int(EPOCHS / 4))
   
    elif SCHEDULER == "custom":
        scheduler_1 = init.get_custom_scheduler(
                              optimizer_1, 
                             int(total_steps * 0.06))
        scheduler_2 = init.get_custom_scheduler(
                              optimizer_2, 
                             int(total_steps * 0.06))
    

    
    ################################################
    ################### Training ###################
    ################################################
    accumulation_step = int(origin_batch / BATCH_SIZE)
    for epoch_i in range(0, EPOCHS):

        model.zero_grad()
        model.train()
        num_batches = 0
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_attention, b_input_label = batch
            b_input_ids = b_input_ids.type(torch.LongTensor).to(config.device)
            b_input_attention = b_input_attention.type(torch.LongTensor).to(config.device)
            b_input_label = b_input_label.type(torch.LongTensor).to(config.device)
            
            #################################################################
            ########################### add token ###########################
            #################################################################
            b_input_ids, b_input_attention, b_input_label = model.insert_token(
                input_ids = b_input_ids, 
                attention_mask = b_input_attention, 
                labels = b_input_label
            )
            #################################################################
            
                
            outputs = model(input_ids = b_input_ids, 
                            attention_mask = b_input_attention, 
                            labels = b_input_label)

            loss = outputs[0].mean() / accumulation_step
            loss.backward()
            num_batches += 1
            
            #accumulation
            if num_batches % accumulation_step == 0 or accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_1.step()
                scheduler_1.step()
                optimizer_2.step()
                scheduler_2.step()
                model.zero_grad()
            
            del b_input_ids
            del b_input_attention
            del b_input_label
            torch.cuda.empty_cache()


 

        if SAVE != -1 and epoch_i % SAVE == 0:
            torch.save(model.state_dict(), SAVE_DIR + '/model_%d_'%(epoch_i) + DATA_NAME)
        valid_path = OUTPUT_DIR + "result/" + DATA_NAME + "[" + str(epoch_i) + "]" + ".txt"
        with open(valid_path, "w") as f:
            f.write('')
        model.eval()

        for batch in valid_dataloader:
            outputs = validation(config, model, batch, TASK_NAME)
            decode_text = tokenizer.batch_decode(outputs[:, config.seq_len + config.rank:],
                                                 skip_special_tokens = True)
            with open(valid_path, "a") as f:
                for text in decode_text:
                    f.write(text.replace("\n", "").strip() + "\n")
                    
            del outputs

            torch.cuda.empty_cache()


    if SAVE != -1:    
        torch.save(model.state_dict(), 
                   SAVE_DIR + '/model_final_' + DATA_NAME)
    
    
if __name__ == "__main__":
    run()
