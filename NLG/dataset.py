"""
from datasets import load_dataset
import torch
import copy
from transformers import GPT2Tokenizer
from tqdm import tqdm
left_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", padding_side = "left")
right_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", padding_side = "right")
data = load_dataset("e2e_nlg")
"""
def train(data):
    sentence = [[], []]
    labels = []
    features = []
    for k in data['train'].features:
        features.append(k)
    
    for i in range(len(data['train'])):
        sentence[0].append(data['train'][i][features[0]].replace("[", " : ").replace("]", ""))
        sentence[1].append(data['train'][i][features[1]])

    left_tokenizer.pad_token = left_tokenizer.eos_token
    right_tokenizer.pad_token = right_tokenizer.eos_token
    train_inputs = left_tokenizer(sentence[0], 
                                 return_tensors='pt', 
                                 max_length = 64,
                                 truncation=True,
                                 padding = 'max_length')
    
    label = right_tokenizer(sentence[1], 
                            return_tensors='pt', 
                            max_length = 64,
                            truncation=True, 
                            padding = 'max_length')
    copy_train_inputs = train_inputs['input_ids'].clone()
    copy_train_label = label['input_ids'].clone()
    
    train_inputs['input_ids'] = torch.cat((train_inputs['input_ids'], 
                                           label['input_ids']), dim = -1)
    train_inputs['attention_mask'] = torch.cat((train_inputs['attention_mask'], 
                                               label['attention_mask']), dim = -1)
    copy_train_inputs[:] = -100
    copy_train_label[torch.where(copy_train_label == 50256)] = -100
    for i in range(len(copy_train_label)):
        flag = False
        for j in range(len(copy_train_label[i])):
            if copy_train_label[i][j] != -100:
                flag = True
            if flag and copy_train_label[i][j] == -100:
                    copy_train_label[i][j] = 50256
                    break
    
    train_inputs['labels'] = torch.cat((copy_train_inputs, copy_train_label),
                                      dim = -1)
    return train_inputs

def valid(data):
    left_tokenizer.pad_token = left_tokenizer.eos_token
    sentence = [[], []]
    labels = []
    features = []
    for k in data['validation'].features:
        features.append(k)
    now = None
    text = ""
    for i in range(len(data['validation'])):
        if now != data['validation'][i][features[0]]:
            sentence[0].append(data['validation'][i][features[0]].replace("[", " : ").replace("]", ""))
            now = data['validation'][i][features[0]]
            if i!= 0:
                text = text + "\n"
        text = text + data['validation'][i][features[1]] + "\n"
        
    left_tokenizer.pad_token = left_tokenizer.eos_token
    train_inputs = left_tokenizer(sentence[0], 
                                 return_tensors='pt',
                                 max_length = 64,
                                 truncation=True, 
                                 padding = 'max_length')
    with open("text.txt", "w") as f:
        f.write(text)
    return train_inputs