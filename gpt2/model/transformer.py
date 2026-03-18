import torch
import torch.nn as nn
from gpt2.model.attn import MultiHeadAttention
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model=768,
            n_heads=12,
            hidden_dim=3072, # normally 4 * d_model
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=n_heads,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, hidden_dim)

    def forward(self, x):
        y = x
        x = self.ln_1(x)
        x = self.attn(x)
        x = x + y
        
        y = x
        x = self.ln_2(x)
        x = self.mlp(x)
        return x + y

class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=True)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=True)

    def forward(self, x):
        # GPT-2 typically uses 'tanh' approximation for GELU, but standard GELU is also widely used
        return self.c_proj(F.gelu(self.c_fc(x)))
