import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt2.model.transformer import TransformerBlock
import math

class GPT2(nn.Module):
    def __init__(
            self,
            vocab_size=50_257,
            max_seq_len=1024,
            d_model=768,
            n_heads=12,
            n_blocks=12,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    hidden_dim=4 * d_model
                )
                for _ in range(n_blocks)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.wte.weight

    def forward(self, x):
        B, T = x.shape
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, block size is only {self.max_seq_len}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=x.device) # shape (T)
        
        tok_emb = self.wte(x) # (B, T, d_model)
        pos_emb = self.wpe(pos) # (T, d_model)
        
        x = tok_emb + pos_emb
        
        for layer in self.transformer_blocks:
            x = layer(x)
            
        x = self.ln_f(x)
        return self.lm_head(x)

    def prefill(self, token_ids, kv_caches):
        """
        token_ids: (B, T) — the full prompt as token IDs
        kv_caches: list of N empty dicts, one per layer
                   after this call, each dict has keys 'k' and 'v'
        Returns:   logits (B, T, vocab_size)
        """
        B, T = token_ids.shape
        pos = torch.arange(0, T, dtype=torch.long, device=token_ids.device)
        
        tok_emb = self.wte(token_ids)                       # (B, T, d_model)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        for i, layer in enumerate(self.transformer_blocks):
            # ── Pre-norm + Attention (with caching) ──
            y = x                                     # save for residual
            x = layer.ln_1(x)                         # (B, T, d_model)
            x = layer.attn._prefill(x, kv_caches[i]) # stores K,V in kv_caches[i]
            x = x + y                                 # residual connection

            # ── Pre-norm + FFN ──
            y = x
            x = layer.ln_2(x)
            x = layer.mlp(x)
            x = x + y                                 # residual connection

        x = self.ln_f(x)                              # (B, T, d_model)
        return self.lm_head(x)                        # (B, T, vocab_size)

    def decode(self, token_ids, kv_caches):
        """
        token_ids: (B, 1) — single new token
        kv_caches: list of N dicts from prefill (or previous decode)
        Returns:   logits (B, 1, vocab_size), updated kv_caches
        """
        B, T = token_ids.shape # T should be 1
        
        # determine position from cache length
        # kv_cache['k'] is (B, seq_len, num_heads, head_dim)
        pos_val = kv_caches[0]['k'].shape[1] 
        pos = torch.tensor([pos_val], dtype=torch.long, device=token_ids.device)
        
        tok_emb = self.wte(token_ids)                 # (B, 1, d_model)
        pos_emb = self.wpe(pos)                       # (1, d_model)
        x = tok_emb + pos_emb                         # (B, 1, d_model)

        for i, layer in enumerate(self.transformer_blocks):
            # ── Pre-norm + Attention (append to cache) ──
            y = x
            x = layer.ln_1(x)
            x, kv_caches[i] = layer.attn.decode(x, kv_caches[i])
            x = x + y

            # ── Pre-norm + FFN ──
            y = x
            x = layer.ln_2(x)
            x = layer.mlp(x)
            x = x + y

        x = self.ln_f(x)
        return self.lm_head(x), kv_caches

    @torch.no_grad()
    def generate(self, token_ids, tokenizer, max_new_tokens=100,
                 temperature=0.1, top_k=50):
        """
        token_ids: (B, T) — prompt token IDs
        tokenizer: tokenizer with .eos_token_id or .eos_id
        Yields:    one token ID (int) at a time
        """
        import torch

        n_layers = len(self.transformer_blocks)

        # Step 1: Create one empty KV cache per layer
        kv_caches = [{} for _ in range(n_layers)]

        # Step 2: PREFILL — process full prompt, fill caches
        logits = self.prefill(token_ids, kv_caches)
        next_logits = logits[:, -1, :] / temperature  # (B, vocab_size)

        # Resolve EOS token ID
        eos_id = getattr(tokenizer, 'eos_token_id', None) or getattr(tokenizer, 'eos_id', None)

        # Step 3: DECODE LOOP — yield one token at a time
        for _ in range(max_new_tokens):
            # ── Sample next token ──
            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < top_vals[:, -1:]] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Decode token ID → text, then yield
            yield tokenizer.decode([next_token.item()])

            # Check for EOS
            if eos_id is not None and (next_token == eos_id).all():
                return
                
            # Avoid out of bounds for positional embedding
            seq_len = kv_caches[0]['k'].shape[1]
            if seq_len >= self.max_seq_len:
                break # out of positional embeddings bounds

            # ── Decode: feed ONLY the new token, grow the cache ──
            logits, kv_caches = self.decode(next_token, kv_caches)
            next_logits = logits[:, -1, :] / temperature
