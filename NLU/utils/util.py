from sklearn.metrics import f1_score, matthews_corrcoef
from scipy import stats
import numpy as np
import torch

class metric:
    def flat_accuracy(preds, labels):
        return np.sum(preds == labels) / len(labels)
    
    #or average = "weighted"
    def F1_score(preds, labels, average = 'binary'):
        try:
            return f1_score(labels, preds, average=average)
        except Exception as e:
            print(e)
            return 0

    def Corr(preds, labels, num = 2):
        try:
            return matthews_corrcoef(labels, preds)
        except Exception as e:
            print(e)
            return 0
    
    def Spear(preds, labels):
        try:
            return stats.spearmanr(preds, labels).correlation
        except Exception as e:
            print(e)
            return 0
    


import datetime
import time
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

class record_training:
    def __init__(self, metric, epochs = 0, data_len = 0, OUTPUT_DIR = "./", file_name = "base"):
        self.before_parameter = {}
        self.total_change = {}
        self.output_dir = OUTPUT_DIR
        self.file_name = file_name
        self.total_loss = 0
        self.t0 = 0
        
        self.metric = metric
        
        self.epochs = epochs
        self.data_len = data_len
        
        with open(OUTPUT_DIR + "parameter_" + file_name + ".txt", 'w') as f:
            f.write('Record parameter changes\n')
        with open(OUTPUT_DIR + "result_" + file_name + ".txt", 'w') as f:
            f.write('Record result\n')

    def save_before_paramter(self, model):
        for name, param in model.named_parameters():
            self.before_parameter[name] = param.clone().detach().requires_grad_(False)
            self.total_change[name] = 0
            
    def save_parameter_change(self, model):
        for name, param in model.named_parameters():
            diff = param.clone().detach().requires_grad_(False) - self.before_parameter[name]
            self.total_change[name] += diff
            
    def record_parameter_change(self, model, epoch_i):
        with open(self.output_dir + "parameter_" + self.file_name + ".txt", 'a') as f:
            f.write('\nEpochs : %d\n'%epoch_i)
        for name, param in model.named_parameters():
            with open(self.output_dir + "parameter_" + self.file_name + ".txt", 'a') as f:
                f.write(name + ' : {}\n'.format(torch.sum(self.total_change[name])))
    
    #새로운 epoch 진입
    def init_epoch(self, epoch_i = 0, total_step = -1):
        if total_step != -1:
            self.total_loss = 0
        self.t0 = time.time()
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write("\n" + '======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, self.epochs) +'Training...\n')
    
    def record_time(self):
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write("\n"+"  Training epcoh took: {:}\n".format(format_time(time.time() - self.t0)))
            
    def save_loss(self, loss):
        self.total_loss += loss
    
    def record_step_loss(self, step, total_step = -1, per_step = 100):
        elapsed = format_time(time.time() - self.t0)        
        if total_step != -1:
            temp_total = self.total_loss / per_step
            self.total_loss = 0
        else:
            temp_total = self.total_loss / step
            self.total_loss = 0
            
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            if total_step == -1:
                f.write('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.\n'.format(step, self.data_len, elapsed))
            else:
                f.write('  Batch {:>5,}.    Elapsed: {:}.\n'.format(total_step, elapsed))
            f.write('Loss / step : {0:.4f}\n'.format(float(temp_total)))
    
    def record_valid_loss(self, step, total_step = -1, per_step = 100):
        elapsed = format_time(time.time() - self.t0)        
        if total_step != -1:
            temp_total = self.total_loss / per_step
            self.total_loss = 0
        else:
            temp_total = self.total_loss / step
            self.total_loss = 0
            
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write('Loss(val) / step : {0:.4f}\n'.format(float(temp_total)))
    
    def record_epoch_loss(self):
        avg_train_loss = self.total_loss / self.data_len
        self.total_loss = 0
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write("\n" + "  Average training loss: {0:.2f}\n".format(float(avg_train_loss)) +"  Training epcoh took: {:}\n".format(format_time(time.time() - self.t0)))
            
            
    def get_score(self, logits, labels, num_labels):
        if num_labels == 1:
            self.metric.add_batch(predictions = logits.flatten(), references = labels.flatten())
        else:
            if len(logits.shape) != 1:
                self.metric.add_batch(predictions = np.argmax(logits, axis=-1).flatten(), 
                                      references = labels.flatten())
            else:
                self.metric.add_batch(predictions = logits, 
                                      references = labels.flatten())
        self.num_labels = num_labels
    
    def record_metric(self):
        score = self.metric.compute()
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            for k in score:
                f.write("  " + k + ": {0:.3f}\n".format(score[k]))
            f.write("  Validation took: {:}\n".format(format_time(time.time() - self.t0)))

            
    def init_validation(self):
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write("\nRunning Validation...\n")
        self.t0 = time.time()



     
