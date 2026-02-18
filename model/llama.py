import torch.nn as nn
import torch.nn.functional as F
from model.transformer import *
from model.attn import *
from model.norm import RMSNorm
import sentencepiece as spm

class LLaMa(nn.Module):
    def __init__(
            self,
            pe_config,
            vocab_size=50_257,
            d_model=2048,
            n_q_heads = 32,
            n_kv_heads = 8,
            n_blocks = 16,
            hidden_dim = 8192,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size,d_model)
        rope = ScaledRoPE(**pe_config)
        self.transformer_blocks = nn.ModuleList(
            [
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
        for layer in self.transformer_blocks:
            x = layer(x)
        x = self.final_norm(x)
        return self.O(x)
    def prefill(self, token_ids, kv_caches):
        """
        token_ids: (B, T) — the full prompt as token IDs
        kv_caches: list of N empty dicts, one per layer
                   after this call, each dict has keys 'k' and 'v'
        Returns:   logits (B, T, vocab_size)
        """
        x = self.emb(token_ids)                       # (B, T, d_model)

        for i, layer in enumerate(self.transformer_blocks):
            # ── Pre-norm + Attention (with caching) ──
            y = x                                     # save for residual
            x = layer.rms1(x)                         # (B, T, d_model)
            x = layer.attn._prefill(x, kv_caches[i]) # stores K,V in kv_caches[i]
            x = x + y                                 # residual connection

            # ── Pre-norm + FFN ──
            y = x
            x = layer.rms2(x)
            x = layer.ffn(x)
            x = x + y                                 # residual connection

        x = self.final_norm(x)                        # (B, T, d_model)
        return self.O(x)                              # (B, T, vocab_size)
    def decode(self, token_ids, kv_caches):
        """
        token_ids: (B, 1) — single new token
        kv_caches: list of N dicts from prefill (or previous decode)
        Returns:   logits (B, 1, vocab_size), updated kv_caches
        """
        x = self.emb(token_ids)                       # (B, 1, d_model)

        for i, layer in enumerate(self.transformer_blocks):
            # ── Pre-norm + Attention (append to cache) ──
            y = x
            x = layer.rms1(x)
            x, kv_caches[i] = layer.attn.decode(x, kv_caches[i])
            x = x + y

            # ── Pre-norm + FFN ──
            y = x
            x = layer.rms2(x)
            x = layer.ffn(x)
            x = x + y

        x = self.final_norm(x)
        return self.O(x), kv_caches

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

            # ── Decode: feed ONLY the new token, grow the cache ──
            logits, kv_caches = self.decode(next_token, kv_caches)
            next_logits = logits[:, -1, :] / temperature