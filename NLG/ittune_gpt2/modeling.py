from .ittune_block import ittune_Block
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Model,
    GPT2LMHeadModel
)
from typing import Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss



class ittuneGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.h = nn.ModuleList([ittune_Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        
    
class ittuneGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = ittuneGPT2Model(config)
        self.seq_len = config.seq_len
        self.rank = config.rank
        self.T_tokens = [35743, 13389, 2420, 3084, 10638, 1366, 6827, 10852] #summarize, transformation, text, table, summary, data, sentence, slot
        self.T_input_ids = torch.Tensor([self.T_tokens[i % len(self.T_tokens)] for i in range(self.rank)]).unsqueeze(dim = 0).long().to(config.device)
        self.T_attention_mask = torch.ones_like(self.T_input_ids).to(config.device)
        
    def insert_token(self,
                     input_ids,
                     attention_mask = None,
                     labels = None,
                    ):
        if input_ids != None and input_ids.shape[-1] >= self.seq_len:
            input_ids = torch.cat((input_ids[:, :self.seq_len], 
                                   self.T_input_ids.expand(input_ids.shape[-2], -1),
                                   input_ids[:, self.seq_len:]),
                                  dim = -1)
        if attention_mask != None:
            attention_mask = torch.cat((attention_mask[:, :self.seq_len], 
                                       self.T_attention_mask.expand(input_ids.shape[-2], -1),
                                       attention_mask[:, self.seq_len:]),
                                       dim = -1)
        if labels != None:
            labels = torch.cat((labels[:, :self.seq_len], 
                                self.T_attention_mask.expand(input_ids.shape[-2], -1) * -100,
                                labels[:, self.seq_len:]),
                                dim = -1)
        return input_ids, attention_mask, labels
        


        