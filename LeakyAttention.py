#liquid attention
import torch 
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math

class LeakyIntegrator(nn.Module):
    def __init__(self, tau=0.5):
        super(LeakyIntegrator, self).__init__()
        self.tau = tau

    def forward(self, x):
        dxdt = -x / self.tau
        return x + dxdt
    

class ConductanceBasedSynapse(nn.Module):
    def __init__(self):
        super(ConductanceBasedSynapse, self).__init__()

    def forward(self, x):
        return F.sigmoid(x)
    


class LeakyIntegrationAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(LeakyIntegrationAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.leaky_integrator = LeakyIntegrator()
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        q = self.query_proj(query)

        k = self.key_prok(key)

        v = self.value_proj(value)

        q = q.view(query.shape[0], query.shape[1], self.nhead, -1).transpose(1, 2)
        k = k.view(key.shape[0], key.shape[1], self.head, -1).transpose(1, 2)
        v = v.view(value.shape[0], value.shape[1], self.nhead, -1).transpose(1, 2)


        q = self.leaky_integrator(q)
        k = self.leaky_integrator(k)

        #attention weights
        attn_weights = torch.matmul(q, k.tranpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.d_model)

        if attn_mask is not None:
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=1)

            attn_weights = self.conductance_based_synapse(attn_weights)

            attn_output = torch.matmul(attn_weights, v)

            attn_output = attn_output.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.d_model)

            attn_output = self.out_proj(attn_output)

            return attn_output, attn_weights
        
class CustomTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu'):
        super(CustomTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = LeakyIntegrator(d_model, nhead)


class CustomTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        def __init__(self, encoder_layer, num_layer, norm=None):
            super(CustomTransformerEncoder, self).__init__(encoder_layer, num_layers, norm)



d_model = 512
nhead = 8
num_layers = 6


encoder_layer = CustomTransformerEncoder(d_model, nhead)
custom_transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers)