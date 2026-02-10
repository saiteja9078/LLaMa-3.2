import torch.nn as nn
import torch.nn.functional as F
from model.transformer import *
from model.attn import *
from model.norm import RMSNorm
import sentencepiece as spm

class Model(nn.Module):
    def __init__(
            self,
            pe_config,
            vocab_size=128_256,
            d_model=2048,
            n_q_heads = 32,
            n_kv_heads = 8,
            n_blocks = 16,
            hidden_dim = 8192,
            
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size,d_model)
        rope = ScaledRoPE(**pe_config)
        self.transformer_blocks = nn.Sequential(
            *[
                Transformer(
                    rope=rope,
                    d_model=d_model,
                    n_q_heads=n_q_heads,
                    n_kv_heads=n_kv_heads,
                    hidden_dim=hidden_dim
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.O = nn.Linear(d_model,vocab_size,bias=False)
        self.O.weight = self.emb.weight
    def forward(self,x):
        x = self.emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.O(x)

