# LLM Inference: The Prefill-Decode Pipeline

## Why Can't We Just Use `forward()` for Generation?

During **training**, we feed in the full sequence at once:

```
Input:  ["The", "cat", "sat", "on"]   →   forward()   →   Logits for all 4 positions
Target: ["cat", "sat", "on", "the"]
```

Every token sees all previous tokens via the causal mask. The full sequence is processed in **one shot** — thats efficient because GPUs love parallelism.

But during **generation**, we don't know the future tokens. We generate **one token at a time**, and each new token depends on ALL previous tokens:

```
Step 1: ["The"]              → forward() → predict "cat"
Step 2: ["The", "cat"]       → forward() → predict "sat"
Step 3: ["The", "cat", "sat"]→ forward() → predict "on"
```

If we naively call `forward()` each time, we **recompute attention over ALL previous tokens** at every step. For a 1000-token sequence, step 999 recomputes attention over 999 tokens — even though tokens 1–998 haven't changed!

> **This is where prefill-decode with KV caching comes in.**

---

## The Two Phases of Inference

```
┌─────────────────────────────────────────────────────┐
│                   INFERENCE                          │
│                                                      │
│   Phase 1: PREFILL              Phase 2: DECODE      │
│   ┌──────────────┐              ┌──────────────┐    │
│   │ Process full  │     ───►    │ Generate one  │    │
│   │ prompt at     │             │ token at a    │    │
│   │ once (parallel)│            │ time (serial) │    │
│   └──────────────┘              └──────┬───────┘    │
│         │                              │             │
│         ▼                              ▼             │
│   Store K,V in cache           Append K,V to cache   │
│   for all prompt tokens        for each new token    │
└─────────────────────────────────────────────────────┘
```

### Phase 1: Prefill

The user gives us a prompt, say: `"The capital of France is"`

We process the **entire prompt** in a single forward pass (just like training). But now, instead of throwing away the intermediate K and V tensors, we **cache them**.

```
Prompt: ["The", "capital", "of", "France", "is"]
         t=0     t=1      t=2    t=3      t=4

For EACH transformer layer:
  1. Compute Q, K, V from the input
  2. Apply RoPE to Q and K
  3. STORE K and V in a per-layer cache ← THIS IS NEW
  4. Run causal attention as normal (Q attends to all K,V)
  5. Residual + FFN as normal
```

After prefill, each layer's cache contains:

```
Layer 0: kv_cache = {
    "k": tensor of shape (B, 5, n_kv_heads, head_dim),  ← K for all 5 prompt tokens
    "v": tensor of shape (B, 5, n_kv_heads, head_dim),  ← V for all 5 prompt tokens
}
Layer 1: kv_cache = { ... same ... }
...
Layer N: kv_cache = { ... same ... }
```

The output logits at position t=4 (last token "is") give us the probability distribution for the **next token**. We sample from it → get `"Paris"`.

---

### Phase 2: Decode (Autoregressive Generation)

Now we need to generate the token after `"Paris"`. We only feed **the single new token** through the model:

```
Input: ["Paris"]   (B, 1, d_model)  ← just ONE token
         t=5
```

For EACH transformer layer:

```
1. Compute Q, K, V from "Paris"
   Q shape: (B, 1, n_q_heads, head_dim)    ← query for ONE position
   K shape: (B, 1, n_kv_heads, head_dim)   ← key for ONE position
   V shape: (B, 1, n_kv_heads, head_dim)   ← value for ONE position

2. Apply RoPE with OFFSET = 5
   ─────────────────────────────────────────
   WHY OFFSET?
   "Paris" is at position 5 in the sequence.
   RoPE encodes ABSOLUTE position via rotation angles.
   Without offset, the model would think "Paris" is at position 0.
   ─────────────────────────────────────────

3. APPEND new K, V to the cache:
   cache["k"] = cat([old_cache_k, new_k], dim=1)
   → shape goes from (B, 5, H, D) to (B, 6, H, D)

4. Attention: Q(1 token) attends to ALL cached K,V (6 tokens)
   ─────────────────────────────────────────
   Q: (B, 1, n_q_heads, head_dim)
   K: (B, 6, n_kv_heads, head_dim)   ← full history from cache
   V: (B, 6, n_kv_heads, head_dim)

   NO CAUSAL MASK needed! Q has only 1 position — it naturally
   can only attend to past tokens (which are all in the cache).
   ─────────────────────────────────────────

5. Residual + FFN as normal
```

We repeat this decode step for each new token:

```
Decode step 1: feed "Paris",   cache grows to 6 tokens → predict "."
Decode step 2: feed ".",       cache grows to 7 tokens → predict "<eos>"
```

---

## Data Flow Through Our Architecture

Lets trace through our exact model, layer by layer.

### Our Architecture Recap

```
Model
├── emb: Embedding(vocab_size, d_model)
├── transformer_blocks: ModuleList of N layers
│   └── Transformer
│       ├── rms1: RMSNorm(d_model)
│       ├── attn: GQFlashAttention
│       │   ├── w_q: Linear(d_model → n_q_heads * head_dim)
│       │   ├── w_k: Linear(d_model → n_kv_heads * head_dim)
│       │   ├── w_v: Linear(d_model → n_kv_heads * head_dim)
│       │   ├── O:   Linear(d_model → d_model)
│       │   └── rope: ScaledRoPE
│       ├── rms2: RMSNorm(d_model)
│       └── ffn: SwiGLUFFN
│           ├── w1: Linear(d_model → hidden_dim)   gate
│           ├── w2: Linear(d_model → hidden_dim)   up
│           └── w3: Linear(hidden_dim → d_model)   down
├── final_norm: RMSNorm(d_model)
└── O: Linear(d_model → vocab_size)    (weight-tied with emb)
```

### Prefill Data Flow (Prompt: 5 tokens, d_model=2048, n_q_heads=32, n_kv_heads=8)

```
token_ids: (B, 5)                          ← integer token IDs
    │
    ▼  emb
x: (B, 5, 2048)                           ← embedded tokens
    │
    ▼  ═══ Transformer Block i ═══
    │
    ├──► y = x                             ← save for residual
    │
    ▼  rms1
x: (B, 5, 2048)                           ← normalized
    │
    ▼  attn._prefill(x, kv_caches[i])
    │   ├── w_q(x) → reshape → Q: (B, 5, 32, 64)
    │   ├── w_k(x) → reshape → K: (B, 5,  8, 64)
    │   ├── w_v(x) → reshape → V: (B, 5,  8, 64)
    │   ├── rope(Q), rope(K)              ← apply positional encoding
    │   ├── kv_caches[i]["k"] = K         ← STORE in cache
    │   ├── kv_caches[i]["v"] = V         ← STORE in cache
    │   ├── attention(Q, K, V, causal=True)
    │   └── O projection → (B, 5, 2048)
    │
    ▼  x = attn_out + y                   ← residual connection
    │
    ├──► y = x                             ← save for residual
    │
    ▼  rms2
x: (B, 5, 2048)
    │
    ▼  ffn
x: (B, 5, 2048)                           ← SwiGLU: w3(silu(w1(x)) * w2(x))
    │
    ▼  x = ffn_out + y                    ← residual connection
    │
    ▼  ═══ ... repeat for all N blocks ═══
    │
    ▼  final_norm
x: (B, 5, 2048)
    │
    ▼  O (output projection, weight-tied)
logits: (B, 5, vocab_size)
    │
    ▼  take logits[:, -1, :]              ← only the LAST position matters
next_token_logits: (B, vocab_size)
```

### Decode Data Flow (Single new token)

```
next_token_id: (B, 1)                     ← the token we just sampled
    │
    ▼  emb
x: (B, 1, 2048)
    │
    ▼  ═══ Transformer Block i ═══
    │
    ├──► y = x
    │
    ▼  rms1 → attn.decode(x, kv_caches[i])
    │   ├── w_q(x) → Q: (B, 1, 32, 64)
    │   ├── w_k(x) → K: (B, 1,  8, 64)
    │   ├── w_v(x) → V: (B, 1,  8, 64)
    │   ├── pos = kv_caches[i]["k"].shape[1]     ← e.g., 5
    │   ├── rope(Q, offset=pos)                  ← position 5
    │   ├── rope(K, offset=pos)
    │   ├── kv_caches[i]["k"] = cat([old_K, K])  ← (B, 6, 8, 64)
    │   ├── kv_caches[i]["v"] = cat([old_V, V])  ← (B, 6, 8, 64)
    │   ├── attention(Q, full_K, full_V, causal=False)
    │   │        ↑(B,1,32,64)  ↑(B,6,8,64)
    │   └── O projection → (B, 1, 2048)
    │
    ▼  residual → rms2 → ffn → residual
    │
    ▼  ═══ ... repeat for all N blocks ═══
    │
    ▼  final_norm → O
logits: (B, 1, vocab_size)
```

---

## The KV Cache: What It Is and Why It Works

### Why Cache K and V (but not Q)?

In attention, the computation is:

```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
```

- **Q** (query): "What am I looking for?" — this changes for every new token
- **K** (key): "What do I contain?" — once computed for a position, **never changes**
- **V** (value): "What do I return?" — once computed for a position, **never changes**

Since K and V for past tokens don't change, we cache them and **only compute new K, V** for the latest token.

### Memory Cost

For each layer, the cache stores:

```
K: (B, seq_len, n_kv_heads, head_dim) → floats
V: (B, seq_len, n_kv_heads, head_dim) → floats
```

With GQA (`n_kv_heads=8` instead of 32), we get **4× memory savings** on the cache compared to MHA!

**Example**: For our model (16 layers, n_kv_heads=8, head_dim=64, fp16):

```
Per token per layer:  2 × 8 × 64 × 2 bytes = 2 KB
Per token all layers: 2 KB × 16 = 32 KB
For 4096 tokens:      32 KB × 4096 = 128 MB per batch element
```

---

## Sampling: From Logits to Tokens

After getting `logits: (B, vocab_size)` from the last position, we need to pick the next token.

### Temperature

```python
logits = logits / temperature
```

- `temperature = 1.0` → raw probabilities (balanced)
- `temperature < 1.0` → sharper distribution (more deterministic)
- `temperature > 1.0` → flatter distribution (more creative / random)

### Top-K Sampling

```python
# Keep only the top-K most probable tokens
top_values, _ = torch.topk(logits, k=50)
logits[logits < top_values[:, -1:]] = -inf   # zero out everything else

probs = softmax(logits)
next_token = multinomial(probs, num_samples=1)
```

This prevents the model from sampling extremely unlikely tokens (gibberish).

---

## The Full `generate()` Loop

Putting it all together, this is the pseudocode for generation:

```
def generate(prompt_tokens, max_new_tokens):

    1. Create empty KV caches (one dict per layer)
       kv_caches = [{} for _ in range(n_layers)]

    2. PREFILL: process entire prompt in one shot
       logits = prefill(prompt_tokens, kv_caches)
                              │
                              ▼
                    Each layer stores its K,V into kv_caches[i]

       next_logits = logits[:, -1, :]    ← prediction for first new token

    3. DECODE LOOP: one token at a time
       for step in range(max_new_tokens):

           a. Sample next token from logits (temperature + top-k)
           b. Append to output sequence

           c. Feed ONLY the new token through every layer:
              logits, kv_caches = decode(new_token, kv_caches)
                                            │
                                            ▼
                                  Each layer appends new K,V
                                  and attends over full cache

           d. next_logits = logits[:, -1, :]

           e. If next_token == EOS: break

    4. Return full generated sequence
```

---

## What We Need to Implement

We need **three methods** on our `Model` class that leverage the existing `_prefill()` and `decode()` methods already defined in `GQFlashAttention`:

| Method                                    | Purpose                              | Calls per layer         |
| ----------------------------------------- | ------------------------------------ | ----------------------- |
| `Model.prefill(token_ids, kv_caches)`     | Process full prompt, populate caches | `layer.attn._prefill()` |
| `Model.decode(token_ids, kv_caches)`      | Process one new token, grow caches   | `layer.attn.decode()`   |
| `Model.generate(prompt, max_tokens, ...)` | Orchestrate prefill → decode loop    | Both above              |

> [!IMPORTANT]
> Since `_prefill` and `decode` are defined on `GQFlashAttention` (the attention submodule), our Model-level methods must **manually** orchestrate the pre-norm → attention → residual → FFN → residual flow for each layer, rather than calling `layer.forward()` directly. This is because `layer.forward()` calls `self.attn.forward()` (the training path) instead of `self.attn._prefill()` or `self.attn.decode()`.

---

## Computational Savings

Here's why this matters — without KV cache vs with:

```
Generating 100 tokens from a 50-token prompt:

WITHOUT cache (naive):
  Step 1:  attention over 51 tokens
  Step 2:  attention over 52 tokens
  ...
  Step 100: attention over 150 tokens
  Total attention operations: Σ(51 to 150) = 15,075 per layer

WITH cache (prefill + decode):
  Prefill:   attention over 50 tokens (once)
  Step 1-100: attention over 1 query × growing cache
  Total: 50 + Σ(51 to 150) = 50 + 10,050 = 10,100 per layer

  But more importantly: each decode step only computes Q,K,V
  for 1 token instead of the full sequence!
```

The real savings are in the **Q, K, V projections** — we go from `O(T × d²)` per step to `O(1 × d²)` per step.
