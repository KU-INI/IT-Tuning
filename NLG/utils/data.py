from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch

    
       
def GPTDataLoader(data, batch = 16, sampler = "random"):
    try:
        data = TensorDataset(data['input_ids'], 
                             data['attention_mask'], 
                             data['labels'])
    except:
        data = TensorDataset(data['input_ids'], 
                             data['attention_mask'])
        
    if sampler == "random":
        data_sampler = RandomSampler(data)
    else:
        data_sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler = data_sampler, batch_size = batch)
    return dataloader
    
def Data(task_name, path, batch_size):
    train = torch.load(path + task_name + "_train")
    valid = torch.load(path + task_name + "_validation")

    train_dataloader = GPTDataLoader(train, batch_size)
    valid_dataloader = GPTDataLoader(valid, 4, sampler = "seq")
    return train_dataloader, valid_dataloader
                

                
                