from .ittune_encoder import ittune_Encoder
import torch

from transformers.models.bert.modeling_bert import (
    BertModel,
    BertForSequenceClassification
)


class ittuneBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ittune_Encoder(config)
        
class ittuneBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = ittuneBertModel(config)