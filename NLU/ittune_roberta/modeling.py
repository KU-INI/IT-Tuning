from .ittune_encoder import ittune_Encoder
import torch

from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaForSequenceClassification
)


class ittuneRobertaModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ittune_Encoder(config)
        
class ittuneRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = ittuneRobertaModel(config)