import torch
import torch.nn as nn
import torch.nn.functional as F

class GQAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_q_heads: int,
            num_kv_heads: int,
            rope: nn.Module
    ):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.group_size = num_q_heads // num_kv_heads

        #projections 
        self.w_q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.O = nn.Linear(d_model,d_model,bias=False)

        self.rope = rope
    def _expand_kv(self,k,v):
        if self.group_size==1:
            return k,v
        k = k.repeat_interleave(self.group_size,dim=1)
        v = v.repeat_interleave(self.group_size,dim=1)
        return k,v
    def forward(self,x: torch.Tensor):
        B, T, _ = x.shape

        #project
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        #Reshape into head dim
        q = q.view(B,T,self.num_q_heads,self.head_dim)
        k = k.view(B,T,self.num_kv_heads,self.head_dim)
        v = v.view(B,T,self.num_kv_heads,self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        out = self._attention(q,k,v,is_causal=True)
        out = out.reshape(B, T, self.d_model)
        return self.O(out)

    def _attention(self,q,k,v,is_causal: bool):
        """
        q: (B, Tq, Hq,  Dh)
        k: (B, Tk, Hkv, Dh)
        v: (B, Tk, Hkv, Dh)
        """
        k,v = self._expand_kv(
            k.transpose(1,2),v.transpose(1,2)
        )
        q = q.transpose(1,2)
        out = F.scaled_dot_product_attention(
            q,k,v,
            attn_mask=None,
            dropout_p=0.,
            is_causal=is_causal
        )
        return out.transpose(1,2)
    def _prefill(self,x,kv_cache):
        """
        x: (B, T, d_model)
        T -> prompt_len
        kv_cache: dict with keys 'k', 'v'
        """
        B, T, _ = x.shape
        q = self.w_q(x).view(B,T,self.num_q_heads,self.head_dim)
        k = self.w_k(x).view(B,T,self.num_kv_heads,self.head_dim)
        v = self.w_v(x).view(B,T,self.num_kv_heads,self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        kv_cache["k"] = k
        kv_cache['v'] = v

        out = self._attention(q,k,v,is_causal=True)
        out = out.reshape(B,T,self.d_model)

        return self.O(out)

    def decode(self,x,kv_cache):
        """
        x: (B, 1, d_model)
        kv_cache: dict with keys 'k', 'v'
        """
        B, _, _ = x.shape
        q = self.w_q(x).view(B,1,self.num_q_heads,self.head_dim)
        k = self.w_k(x).view(B,1,self.num_kv_heads,self.head_dim)
        v = self.w_v(x).view(B,1,self.num_kv_heads,self.head_dim)

        pos = kv_cache['k'].shape[1]
        q = self.rope(q,offset = pos)
        k = self.rope(k,offset = pos)

        kv_cache['k'] = torch.cat([kv_cache['k'],k],dim=1)
        kv_cache['v'] = torch.cat([kv_cache['v'],v],dim=1)

        out = self._attention(
            q,kv_cache['k'],kv_cache['v'],is_causal=False
        )
        out = out.reshape(B,1,self.d_model)
        return self.O(out), kv_cache
    
class ScaledRoPE(nn.Module):
    def __init__(
        self,
        d: int,
        base: float = 500_000.,
        scaling_factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.base = base
        self.d = d
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        
        self.cos_cached = None
        self.sin_cached = None
        
        # Compute scaled frequencies once at initialization
        self.register_buffer("inv_freq", self._compute_scaled_inv_freq())
    
    def _compute_scaled_inv_freq(self):
        """
        Compute inverse frequencies with Llama 3.2-style scaling.
        Different frequency components get different scaling factors.
        """
        # Step 1: Compute base inverse frequencies (same as before)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.d, 2, dtype=torch.float32) / self.d)
        )
        # Shape: (d/2,)
        # Example for d=64: [1.0, 0.795, 0.632, ..., 0.0000025]
        
        # Step 2: Calculate "wavelength" for each frequency
        # Wavelength = how many positions before the pattern repeats
        # wavelength = 2π / frequency = 2π * inv_freq
        wavelengths = 2 * torch.pi / inv_freq
        # Example: [6.28, 7.91, 9.94, ..., 2,513,274]
        
        # Step 3: Define wavelength boundaries
        low_freq_wavelen = self.original_max_position_embeddings / self.low_freq_factor
        high_freq_wavelen = self.original_max_position_embeddings / self.high_freq_factor
        # low_freq_wavelen = 8192 / 1.0 = 8192
        # high_freq_wavelen = 8192 / 4.0 = 2048
        
        # Step 4: Compute scaling factor for each frequency component
        new_inv_freq = []
        
        for i, (freq, wavelen) in enumerate(zip(inv_freq, wavelengths)):
            if wavelen < high_freq_wavelen:
                # HIGH FREQUENCY (short wavelength < 2048)
                # Pattern repeats quickly - good for local attention
                # No scaling needed
                scale = 1.0
                
            elif wavelen > low_freq_wavelen:
                # LOW FREQUENCY (long wavelength > 8192)
                # Pattern repeats slowly - good for long-range attention
                # Needs full scaling to extend context
                scale = self.scaling_factor  # 8.0
                
            else:
                # MEDIUM FREQUENCY (2048 ≤ wavelength ≤ 8192)
                # Smooth interpolation between no scaling and full scaling
                
                # Calculate where we are in the range [2048, 8192]
                # smooth goes from 0.0 to 1.0
                smooth = (
                    (self.original_max_position_embeddings / wavelen - self.high_freq_factor)
                    / (self.low_freq_factor - self.high_freq_factor)
                )
                # Example for wavelen=4096:
                # smooth = (8192/4096 - 4.0) / (1.0 - 4.0)
                #        = (2.0 - 4.0) / (-3.0)
                #        = 0.667
                
                # Interpolate scale from 1.0 to 8.0
                scale = 1.0 + smooth * (self.scaling_factor - 1.0)
                # scale = 1.0 + 0.667 * 7.0 = 5.67
            
            # Apply scaling by dividing the frequency
            # Lower frequency = slower rotation = longer effective wavelength
            new_inv_freq.append(freq / scale)
        
        return torch.tensor(new_inv_freq, dtype=torch.float32)
    
    def _build_cache(self, seq_len: int, device, dtype):
        if (
            self.cos_cached is not None
            and seq_len <= self.cos_cached.shape[1]
            and self.cos_cached.device == device
            and self.cos_cached.dtype == dtype
        ):
            return

        inv_freq = self.inv_freq.to(device=device, dtype=dtype)

        seq_idx = torch.arange(seq_len, device=device, dtype=dtype)

        idx_theta = seq_idx[:, None] * inv_freq[None, :]
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()[None, :, None, :]
        self.sin_cached = idx_theta2.sin()[None, :, None, :]
        def _neg_half(self, x_rope):
            """Helper for rotation"""
            d_by_2 = self.d // 2
            return torch.cat([-x_rope[..., d_by_2:], x_rope[..., :d_by_2]], dim=-1)
    def forward(self, x, offset: int = 0):
        seq_len = x.shape[1] + offset
        self._build_cache(seq_len, x.device, x.dtype)

        T = x.shape[1]
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        neg_half_x = self._neg_half(x_rope)

        cos = self.cos_cached[:, offset:offset+T]
        sin = self.sin_cached[:, offset:offset+T]

        x_rope = (x_rope * cos) + (neg_half_x * sin)

        return torch.cat([x_rope, x_pass], dim=-1)
        
    def _neg_half(self, x_rope):
        """Helper for rotation"""
        d_by_2 = self.d // 2
        return torch.cat([-x_rope[..., d_by_2:], x_rope[..., :d_by_2]], dim=-1)