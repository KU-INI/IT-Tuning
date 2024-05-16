from torch import nn
import torch
import math
from typing import List, Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block,
)
import copy

class ittune_Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.config = config
        self.seq_len = config.seq_len
        self.rank = self.config.rank
        self.seq_end = config.seq_len + self.config.rank
        
        
        
        ####################################################################
        #########IT-Token이 서로를 attend하지 않도록 제한####################
        ####################################################################
        temp_mask_under = torch.tril(torch.ones((config.max_position_embeddings, config.max_position_embeddings), dtype=torch.bool), diagonal = -1).view(
                1, 1, config.max_position_embeddings, config.max_position_embeddings
            )
        temp_mask_upper = torch.triu(torch.ones((config.max_position_embeddings, config.max_position_embeddings), dtype=torch.bool), diagonal = 1).view(
                1, 1, config.max_position_embeddings, config.max_position_embeddings
            )

        self.bias[:, :, :, self.seq_len:self.seq_end] = True
        self.bias[:, :, self.seq_len:self.seq_end, self.seq_len:self.seq_end][torch.where(temp_mask_under[:, :, self.seq_len:self.seq_end, self.seq_len:self.seq_end] == True)]= False
        self.bias[:, :, self.seq_len:self.seq_end, self.seq_len:self.seq_end][torch.where(temp_mask_upper[:, :, self.seq_len:self.seq_end, self.seq_len:self.seq_end] == True)]= False
        
        self.register_buffer("t_mask", copy.deepcopy(self.bias))
        ####################################################################
        
        self.query_vector_0 = nn.Parameter(torch.ones(self.rank, config.hidden_size))
        self.query_vector_1 = nn.Parameter(torch.ones(config.hidden_size))
    
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            
            ####################################################################
            #########IT-Token이 서로를 attend하지 않도록 제한####################
            ####################################################################
            causal_mask = self.t_mask[:, :, key_length - query_length : key_length, :key_length]
            
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
            
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        temp_query = ()
        ############################   ittune   ##########################
        if query.shape[-2] >= self.rank:
            q0 = query[:, self.seq_len:self.seq_end].mul(self.query_vector_0)
            q1 = query[:, :self.seq_len].mul(self.query_vector_1)
            query = torch.cat((q1, q0, query[:, self.seq_end:]), dim = -2)
        elif query.shape[-2] == 1:
            pass
        else:
            raise Exception("check sequence len")
        #################################################################
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            #print(query.shape)
            #print(attention_mask.shape)
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)        

    
    
class ittune_Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.config = config
        self.seq_len = config.seq_len
        self.rank = self.config.rank
        self.seq_end = config.seq_len + self.config.rank
        
        self.attn = ittune_Attention(config, layer_idx=layer_idx)
        if config.add_cross_attention:
            self.crossattention = ittune_Attention(config, is_cross_attention=True, layer_idx=layer_idx)
        
        self.attention_vector_0 = nn.Parameter(torch.zeros(self.rank, config.hidden_size))
        self.attention_vector_1 = nn.Parameter(torch.zeros(config.hidden_size))
            
        self.hidden_vector_0 = nn.Parameter(torch.ones(self.rank, config.hidden_size))
        self.hidden_vector_1 = nn.Parameter(torch.ones(config.hidden_size))
        
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual
        
        
        
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights
            

        ###############################################################################################
        if hidden_states.shape[-2] >= self.rank:
            hidden_states[:, self.seq_len:self.seq_end] = torch.add(hidden_states[:, self.seq_len:self.seq_end], self.attention_vector_0)
            hidden_states[:, :self.seq_len] = torch.add(hidden_states[:, :self.seq_len], self.attention_vector_1)
        elif hidden_states.shape[-2] == 1:
            pass
        else:
            raise Exception("check sequence len") 
        ###############################################################################################
        
        
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
       
       
        temp_hidden = ()
        ###############################################################################################
        if hidden_states.shape[-2] >= self.rank:
            h0 = hidden_states[:, self.seq_len:self.seq_end].mul(self.hidden_vector_0)
            h1 = hidden_states[:, :self.seq_len].mul(self.hidden_vector_1)
            hidden_states = torch.cat((h1, h0, hidden_states[:, self.seq_end:]), dim = -2)
        elif hidden_states.shape[-2] == 1:
            pass
        else:
            raise Exception("check sequence len")
        ###############################################################################################
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)