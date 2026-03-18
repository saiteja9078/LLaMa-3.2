import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projections 
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=True)
        self.c_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape

        # project to q, k, v
        qkv = self.c_attn(x)
        # Split into q, k, v
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape into head dim
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        out = self._attention(q, k, v, is_causal=True)
        out = out.reshape(B, T, self.d_model)
        return self.c_proj(out)

    def _attention(self, q, k, v, is_causal: bool):
        """
        q: (B, Tq, H, Dh)
        k: (B, Tk, H, Dh)
        v: (B, Tk, H, Dh)
        """
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.,
            is_causal=is_causal
        )
        return out.transpose(1, 2)

    def _prefill(self, x, kv_cache):
        """
        x: (B, T, d_model)
        T -> prompt_len
        kv_cache: dict with keys 'k', 'v'
        """
        B, T, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        kv_cache["k"] = k
        kv_cache['v'] = v

        out = self._attention(q, k, v, is_causal=True)
        out = out.reshape(B, T, self.d_model)

        return self.c_proj(out)

    def decode(self, x, kv_cache):
        """
        x: (B, 1, d_model)
        kv_cache: dict with keys 'k', 'v'
        """
        B, _, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        q = q.view(B, 1, self.num_heads, self.head_dim)
        k = k.view(B, 1, self.num_heads, self.head_dim)
        v = v.view(B, 1, self.num_heads, self.head_dim)

        kv_cache['k'] = torch.cat([kv_cache['k'], k], dim=1)
        kv_cache['v'] = torch.cat([kv_cache['v'], v], dim=1)

        out = self._attention(
            q, kv_cache['k'], kv_cache['v'], is_causal=False
        )
        out = out.reshape(B, 1, self.d_model)
        return self.c_proj(out), kv_cache
