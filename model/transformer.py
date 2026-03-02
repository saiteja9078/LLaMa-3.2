import torch
import torch.nn as nn
from model.attn import *
from model.norm import RMSNorm
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(
            self,
            rope,
            d_model = 2048,
            n_q_heads = 32,
            n_kv_heads = 8,
            hidden_dim = 8192,
    ):
        super().__init__()
        self.rms1 = RMSNorm(d_model)
        self.attn = GQAttention(
            d_model=d_model,
            num_q_heads=n_q_heads,
            num_kv_heads=n_kv_heads,
            rope = rope
        )
        self.rms2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(
            d_model,hidden_dim
        )
    def forward(self,x):
        y = x
        x = self.rms1(x)
        x = self.attn(x)
        x = x+y
        y = x
        x = self.rms2(x)
        x = self.ffn(x)
        return x+y

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model , hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(d_model,hidden_dim,bias=False)
        self.w2 = nn.Linear(d_model,hidden_dim,bias=False)
        self.w3 = nn.Linear(hidden_dim,d_model,bias=False)

    def forward(self,x):
        return self.w3(
            F.silu(self.w1(x)) * self.w2(x)
        )