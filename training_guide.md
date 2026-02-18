# LLM Training: From Raw Text to Trained Weights

## What Are We Building?

We're going to train our Llama-style model on the **FineWeb-Edu** dataset — a curated collection of high-quality educational web content. Here's the big picture:

```
Raw Text (FineWeb-Edu)
        │
        ▼
  GPT-2 Tokenizer      → token IDs
        │
        ▼
  Token Packing         → fixed-length chunks of 1024 tokens
        │
        ▼
  Next-Token Prediction → shift input by 1 to get labels
        │
        ▼
  Forward Pass           → model predicts next token at every position
        │
        ▼
  Cross-Entropy Loss     → how wrong were the predictions?
        │
        ▼
  Backprop + AdamW       → update weights
        │
        ▼
  Repeat for billions of tokens...
```

The training must be **resumable** — after training on 10B tokens, we can checkpoint and continue training on another 10–20B tokens with the same weights.

---

## Part 1: The Dataset

### Why Do We Need a Custom Dataset?

FineWeb-Edu has **10 billion tokens** worth of text. We can't download it all at once. Instead, we **stream** it — one document at a time, never keeping more than a few in memory.

Also, raw documents have varying lengths. But our model expects fixed-length sequences. So we need to:

1. Stream documents from HuggingFace
2. Tokenize each document
3. Pack tokens into fixed-length chunks (no padding, no waste)

### Step 1: Loading the Data

```python
from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True  # streams data, doesn't download all at once
)
```

Each element of `dataset` is a dict with a `"text"` field:

```python
# Example:
# {
#     "text": "Photosynthesis is the process by which plants convert...",
#     "url": "https://...",
#     "token_count": 1523,
#     ...
# }
```

We only care about `"text"`.

---

### Step 2: Tokenization

We use the GPT-2 tokenizer (same one our model's `vocab_size=50_257` was built for):

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

Tokenizing a document:

```python
tokens = tokenizer.encode("Photosynthesis is the process...")
# → [6158, 38546, 271, 318, 262, 1429, ...]
```

> [!IMPORTANT]
> We add the **EOS token** (`tokenizer.eos_token_id = 50256`) between documents. This tells the model "this document ended, a new one is starting." Without this, the model would try to connect unrelated documents.

```python
tokens = tokenizer.encode(doc["text"]) + [tokenizer.eos_token_id]
```

---

### Step 3: Token Packing (The Key Trick)

Instead of treating each document as a separate sample (which would require padding short docs and truncating long ones), we **concatenate all documents into one endless stream** and chop it into fixed-length windows:

```
Document 1 tokens: [A, B, C, D, E, <eos>]
Document 2 tokens: [F, G, H, <eos>]
Document 3 tokens: [I, J, K, L, M, N, O, P, <eos>]

Concatenated stream:
[A, B, C, D, E, <eos>, F, G, H, <eos>, I, J, K, L, M, N, O, P, <eos>, ...]

Packed into chunks of seq_len=6:
Chunk 1: [A, B, C, D, E, <eos>]
Chunk 2: [F, G, H, <eos>, I, J]
Chunk 3: [K, L, M, N, O, P]
          ↑ documents can span chunks — that's fine!
```

**Why this works:**

- **No padding waste** — every token is meaningful
- **No truncation** — long documents simply span multiple chunks
- **Simple** — just fill a buffer and drain it

---

### Step 4: The IterableDataset Class

Here's the full dataset class. Let's build it piece by piece.

**The skeleton:**

```python
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

class FineWebEduDataset(IterableDataset):
    def __init__(self, seq_len=1024, split="train"):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split=split,
            streaming=True
        )
```

> [!NOTE]
> We use `IterableDataset` (not `Dataset`) because we're streaming. With `IterableDataset`, you implement `__iter__` instead of `__getitem__` + `__len__`.

**The iterator — the heart of the class:**

```python
    def __iter__(self):
        buffer = []  # accumulate tokens here

        for doc in self.dataset:
            # Tokenize + append EOS
            tokens = self.tokenizer.encode(doc["text"]) + [self.tokenizer.eos_token_id]
            buffer.extend(tokens)

            # Drain buffer into chunks whenever we have enough
            while len(buffer) >= self.seq_len + 1:
                # +1 because we need seq_len tokens for input
                # and the same seq_len tokens shifted by 1 for labels
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)   # input:  positions 0..seq_len-1
                y = torch.tensor(chunk[1:],  dtype=torch.long)   # labels: positions 1..seq_len

                yield x, y
```

Let's visualize what `x` and `y` look like:

```
chunk:  [A,  B,  C,  D,  E,  F,  G]    (seq_len + 1 = 7 tokens)

x:      [A,  B,  C,  D,  E,  F]        (input:  first 6 tokens)
y:      [B,  C,  D,  E,  F,  G]        (labels: last 6 tokens)

Position 0: input A, predict B  ✓
Position 1: input B, predict C  ✓
Position 2: input C, predict D  ✓
...
```

The model sees `x`, produces logits at every position, and we compare with `y` using cross-entropy loss. This is **next-token prediction** — the foundation of all LLM training.

**Complete class:**

```python
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

class FineWebEduDataset(IterableDataset):
    def __init__(self, seq_len=1024, split="train"):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split=split,
            streaming=True
        )

    def __iter__(self):
        buffer = []

        for doc in self.dataset:
            tokens = self.tokenizer.encode(doc["text"]) + [self.tokenizer.eos_token_id]
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)

                yield x, y
```

Save this as **`dataset.py`** in your project root.

---

## Part 2: The Training Loop

### Overview

The training loop has these components:

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                            │
│                                                              │
│  1. Model + Optimizer + LR Scheduler                        │
│  2. For each batch:                                          │
│     a. Forward pass    → logits                              │
│     b. Loss            → cross-entropy(logits, labels)       │
│     c. Backward pass   → gradients                           │
│     d. Gradient accum  → accumulate over N micro-batches     │
│     e. Optimizer step  → update weights                      │
│     f. Log progress    → loss, lr, tokens/sec                │
│  3. Periodically save checkpoints (model + optimizer + step) │
│  4. Resume from checkpoint when continuing training          │
└─────────────────────────────────────────────────────────────┘
```

### Step 1: Configuration

Put all hyperparameters in one place so they're easy to tune:

```python
import torch
import torch.nn as nn
import time
import os
import math

# ─── Hyperparameters ───
config = {
    # Model
    "vocab_size": 50_257,
    "d_model": 2048,
    "n_q_heads": 32,
    "n_kv_heads": 8,
    "n_blocks": 16,
    "hidden_dim": 8192,

    # RoPE
    "pe_config": {
        "d": 64,                  # head_dim = d_model / n_q_heads = 2048/32 = 64
        "base": 500_000.0,
        "scaling_factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },

    # Training
    "seq_len": 1024,
    "batch_size": 8,              # micro-batch size (per step)
    "grad_accum_steps": 8,        # effective batch = 8 * 8 = 64 sequences
                                  # = 64 * 1024 = 65,536 tokens per update
    "max_steps": 150_000,         # total optimizer steps
    "learning_rate": 3e-4,
    "min_lr": 3e-5,               # 10% of max LR
    "warmup_steps": 2000,
    "weight_decay": 0.1,
    "grad_clip": 1.0,

    # Checkpointing
    "checkpoint_dir": "checkpoints",
    "save_every": 5000,           # save checkpoint every N steps
    "resume_from": None,          # path to checkpoint to resume from
                                  # e.g. "checkpoints/step_10000.pt"
}
```

> [!NOTE]
> **Effective batch size** = `batch_size × grad_accum_steps × seq_len` tokens per optimizer step. With the defaults above: `8 × 8 × 1024 = 65,536 tokens/step`. Over 150,000 steps that's ~10B tokens — matching our dataset size.

---

### Step 2: Learning Rate Schedule

Modern LLMs use a **cosine decay** schedule with **linear warmup**:

```
LR
 │
 │  ╱‾‾‾‾‾‾‾‾‾‾‾‾╲
 │ ╱                ╲
 │╱                  ╲
 │                    ╲___________
 │
 └─────────────────────────────────→ steps
   warmup     cosine decay      min_lr
```

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # 1) Linear warmup
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    # 2) After max_steps, return min_lr
    if step >= max_steps:
        return min_lr

    # 3) Cosine decay between warmup_steps and max_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # goes from 1 → 0
    return min_lr + coeff * (max_lr - min_lr)
```

**Why warmup?** At the start, weights are random and gradients are noisy. A large LR would cause unstable, destructive updates. We ramp up gradually so the optimizer can "find its footing."

**Why cosine decay?** As training progresses, we want finer adjustments. The cosine schedule smoothly reduces the LR, avoiding the sharp drops of step-based schedules.

**Why min_lr > 0?** A completely zero LR means NO learning. We keep a small `min_lr` (typically 10% of `max_lr`) so the model can still make tiny corrections at the end and also supports continuing training beyond `max_steps`.

---

### Step 3: Setting Up Model, Optimizer, and DataLoader

```python
from model.llama import Model
from dataset import FineWebEduDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ─── Model ───
model = Model(
    pe_config=config["pe_config"],
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    n_q_heads=config["n_q_heads"],
    n_kv_heads=config["n_kv_heads"],
    n_blocks=config["n_blocks"],
    hidden_dim=config["hidden_dim"],
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**The optimizer — AdamW with weight decay:**

```python
# ─── Optimizer ───
# Separate parameters: apply weight decay to weights, NOT to biases/norms
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if param.dim() >= 2:       # weight matrices (2D+)
        decay_params.append(param)
    else:                      # biases, LayerNorm/RMSNorm scales (1D)
        no_decay_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": decay_params,    "weight_decay": config["weight_decay"]},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=config["learning_rate"], betas=(0.9, 0.95), fused=True)
```

> [!IMPORTANT]
> **Why separate decay and no-decay groups?**
>
> Weight decay is a regularizer — it penalizes large weights by slightly shrinking them each step. This makes sense for weight matrices (prevents them from growing too large). But for bias terms and normalization scales, decay would fight against their purpose. The RMSNorm `scale` parameter needs freedom to be whatever value normalizes best.
>
> **Why `betas=(0.9, 0.95)`?** Standard is `(0.9, 0.999)`, but LLM training literature (GPT-3, Chinchilla, Llama) found `β2=0.95` works better — it gives less weight to very old squared-gradient history, adapting faster to the changing loss landscape during pre-training.
>
> **Why `fused=True`?** This is a PyTorch optimization that fuses the Adam computation into a single GPU kernel, making the optimizer step ~20% faster. Only available on CUDA.

**The DataLoader:**

```python
# ─── DataLoader ───
train_dataset = FineWebEduDataset(seq_len=config["seq_len"])

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    num_workers=2,
    pin_memory=True,
    prefetch_factor=4,
)
```

> [!NOTE]
> `pin_memory=True` allocates the CPU tensors in page-locked (pinned) memory, which speeds up the CPU→GPU transfer. `prefetch_factor=4` tells the DataLoader to fetch 4 batches ahead of time per worker, keeping the GPU fed while we tokenize.
>
> **num_workers**: For streaming datasets, each worker gets its own **independent** copy of the dataset stream. With `num_workers=2`, two workers will stream **the same data** in parallel (both start from the beginning). This means you'll see **duplicate data**. If you want to avoid this, you can either:
>
> - Shard the dataset manually (give each worker a different slice)
> - Use `num_workers=0` for no parallelism (simpler, slightly slower tokenization)
>
> For pre-training where you see each token only once anyway, `num_workers=0` is the safest choice. If you want to use multiple workers, see the "Multi-Worker Sharding" section at the end.

---

### Step 4: Checkpointing (Save & Resume)

This is what enables **resumable training**. We save everything needed to continue exactly where we left off:

```python
def save_checkpoint(model, optimizer, step, tokens_seen, config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config,
    }
    torch.save(checkpoint, path)
    print(f"  💾 Checkpoint saved: {path} (step {step}, {tokens_seen:,} tokens)")
```

```python
def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    tokens_seen = checkpoint["tokens_seen"]
    print(f"  ✅ Resumed from {path} (step {step}, {tokens_seen:,} tokens)")
    return step, tokens_seen
```

**What we save and why:**

| Saved item             | Why?                                                                                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_state_dict`     | The trained weights — the whole point                                                                                                                                                 |
| `optimizer_state_dict` | AdamW keeps running averages (momentum, variance) per parameter. Without these, resuming would be like starting optimization from scratch with warm weights — causing a spike in loss |
| `step`                 | So the LR scheduler picks up at the right learning rate                                                                                                                               |
| `tokens_seen`          | So we know how many tokens to skip in the dataset when resuming                                                                                                                       |
| `config`               | For reference — so you know what hyperparams this checkpoint used                                                                                                                     |

> [!IMPORTANT]
> **For resuming on a new dataset (e.g., another 10–20B tokens):** You load the `model_state_dict` and `optimizer_state_dict`, but **reset `step` to 0** and set a new `max_steps`. This way the LR schedule restarts (warmup again from `min_lr` to `max_lr`, then cosine decay). The model keeps its learned weights, the optimizer keeps its momentum — but the LR gets a fresh schedule for the new training run.

```python
# To continue training on NEW data with existing weights:
def load_checkpoint_for_continued_training(path, model, optimizer, device):
    """Load weights + optimizer state, but reset step counter."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    total_tokens_so_far = checkpoint["tokens_seen"]
    print(f"  ✅ Loaded weights from {path} ({total_tokens_so_far:,} tokens trained)")
    print(f"  🔄 Step reset to 0 — starting fresh LR schedule for continued training")
    return 0, total_tokens_so_far  # step=0, but keep total token count for logging
```

---

### Step 5: The Training Loop

This is where everything comes together:

```python
def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ─── Setup ───────────────────────────────────────────────
    model = Model(
        pe_config=config["pe_config"],
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_q_heads=config["n_q_heads"],
        n_kv_heads=config["n_kv_heads"],
        n_blocks=config["n_blocks"],
        hidden_dim=config["hidden_dim"],
    ).to(device)

    # Separate weight-decay vs no-decay params
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": config["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config["learning_rate"], betas=(0.9, 0.95), fused=(device == "cuda"))

    # ─── Resume from checkpoint if specified ─────────────────
    step = 0
    tokens_seen = 0

    if config["resume_from"]:
        step, tokens_seen = load_checkpoint(
            config["resume_from"], model, optimizer, device
        )

    # ─── Data ────────────────────────────────────────────────
    train_dataset = FineWebEduDataset(seq_len=config["seq_len"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"])
    data_iter = iter(train_loader)

    # If resuming, skip ahead to where we left off
    if tokens_seen > 0:
        tokens_to_skip = tokens_seen // config["seq_len"]
        print(f"  ⏩ Skipping {tokens_to_skip:,} chunks to resume position...")
        for i, _ in enumerate(data_iter):
            if i >= tokens_to_skip:
                break

    # ─── Compile (optional but ~2x faster on modern GPUs) ───
    if device == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("  ⚡ Model compiled with torch.compile")

    # ─── Training Loop ──────────────────────────────────────
    model.train()
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    t0 = time.time()

    while step < config["max_steps"]:
        optimizer.zero_grad()
        loss_accum = 0.0

        # ── Gradient Accumulation Loop ──
        for micro_step in range(config["grad_accum_steps"]):
            # Get next batch (restart iterator if dataset ends)
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            # Forward with mixed precision
            with torch.autocast(device_type=device, dtype=dtype):
                logits = model(x)                      # (B, T, vocab_size)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
                    y.view(-1)                          # (B*T,)
                )
                # Scale loss by number of accumulation steps
                loss = loss / config["grad_accum_steps"]

            loss.backward()
            loss_accum += loss.item()
            tokens_seen += x.numel()

        # ── Gradient Clipping ──
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        # ── Update LR ──
        lr = get_lr(step, config["warmup_steps"], config["max_steps"],
                    config["learning_rate"], config["min_lr"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Optimizer Step ──
        optimizer.step()
        step += 1

        # ── Logging ──
        if step % 100 == 0 or step == 1:
            dt = time.time() - t0
            tokens_per_sec = tokens_seen / dt if dt > 0 else 0
            print(
                f"step {step:>6d} | "
                f"loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | "
                f"tokens {tokens_seen:>12,} | "
                f"tok/s {tokens_per_sec:,.0f}"
            )

        # ── Checkpointing ──
        if step % config["save_every"] == 0:
            save_checkpoint(
                model, optimizer, step, tokens_seen, config,
                os.path.join(config["checkpoint_dir"], f"step_{step}.pt")
            )

    # ── Final checkpoint ──
    save_checkpoint(
        model, optimizer, step, tokens_seen, config,
        os.path.join(config["checkpoint_dir"], f"step_{step}_final.pt")
    )
    print(f"\n🎉 Training complete! {tokens_seen:,} tokens processed in {step} steps.")
```

---

### Step 6: Let's Trace Through One Training Step

This is what happens during a **single optimizer step** (with `grad_accum_steps=8`):

```
optimizer.zero_grad()          ← reset all gradients to 0

  ╔══════════════════════════ Micro-step 1/8 ══════════════════════╗
  ║  x, y = next(data_iter)        (B=8 sequences, T=1024 tokens) ║
  ║  x.shape = (8, 1024)           y.shape = (8, 1024)            ║
  ║                                                                ║
  ║  logits = model(x)             (8, 1024, 50257) — predictions  ║
  ║                                                                ║
  ║  loss = cross_entropy(logits, y)                               ║
  ║  loss /= 8     ← divide by accum_steps so gradients average   ║
  ║                                                                ║
  ║  loss.backward()   ← gradients ACCUMULATE (not replaced)      ║
  ╚════════════════════════════════════════════════════════════════╝

  ╔══════════════════════════ Micro-step 2/8 ══════════════════════╗
  ║  x, y = next(data_iter)        (another batch of 8 sequences) ║
  ║  loss = cross_entropy(...) / 8                                 ║
  ║  loss.backward()               ← adds to existing gradients   ║
  ╚════════════════════════════════════════════════════════════════╝

  ... repeat for micro-steps 3–8 ...

After 8 micro-steps, each parameter's .grad is the AVERAGE over
8 × 8 = 64 sequences = 65,536 tokens

clip_grad_norm_(params, 1.0)   ← prevent gradient explosion
set LR based on schedule
optimizer.step()                ← actually update the weights
```

**Why gradient accumulation?** Training a 2048-dim model with batch_size=64 on seq_len=1024 would need ~40GB+ GPU memory. By accumulating gradients over 8 micro-batches of size 8, we get the same mathematical result using only ~5GB per micro-batch.

---

### Step 7: Mixed Precision — What `torch.autocast` Does

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x)
    loss = cross_entropy(logits, y)
```

This wraps the forward pass in **mixed precision**:

```
Without autocast:
  All matrix multiplications in float32 (32 bits per number)
  Memory: 4 bytes per parameter
  Speed:  baseline

With autocast (bfloat16):
  Matrix multiplications in bfloat16 (16 bits per number)
  Loss computation in float32 (for numerical stability)
  Memory: ~2 bytes per activation (50% less VRAM for activations)
  Speed:  ~2x faster on modern GPUs (A100, 4090, etc.)
```

> [!NOTE]
> **bfloat16 vs float16**: bfloat16 has the same exponent range as float32 (so no overflow issues) but less precision. float16 has more precision but limited range — it needs a `GradScaler` to prevent underflow. bfloat16 doesn't need a scaler, making it simpler. **Use bfloat16 if your GPU supports it** (Ampere+), otherwise fall back to float16 with `torch.amp.GradScaler`.

---

## Part 3: The Entry Point

Save all the above as **`train.py`** and add this at the bottom:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--continue_training", type=str, default=None,
                        help="Path to checkpoint to continue training (reset step, keep weights)")
    parser.add_argument("--max_steps", type=int, default=150_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    args = parser.parse_args()

    # Update config
    config["max_steps"] = args.max_steps
    config["batch_size"] = args.batch_size
    config["seq_len"] = args.seq_len

    if args.resume:
        config["resume_from"] = args.resume
    elif args.continue_training:
        config["resume_from"] = args.continue_training
        config["continue_mode"] = True  # signal to reset step counter

    train(config)
```

---

## Part 4: Running Training

### First training run (10BT)

```bash
python train.py
```

This will:

- Stream FineWeb-Edu 10BT
- Train for 150,000 steps (~10B tokens)
- Save checkpoints every 5,000 steps to `checkpoints/`

### Resume if interrupted

```bash
python train.py --resume checkpoints/step_75000.pt
```

This loads the exact state at step 75,000 and continues from there — same LR schedule, same optimizer state, skips past the data already seen.

### Continue training on another 10–20B tokens

```bash
python train.py --continue_training checkpoints/step_150000_final.pt --max_steps 150000
```

This:

- Loads the trained weights and optimizer momentum
- Resets step to 0 (fresh LR schedule with warmup)
- Trains for another 150,000 steps on the data

> [!TIP]
> For continued training, you might want to use a different dataset split or a larger dataset. Just modify the `FineWebEduDataset` to accept a dataset name/path as a parameter:
>
> ```python
> class FineWebEduDataset(IterableDataset):
>     def __init__(self, seq_len=1024, dataset_name="sample-10BT"):
>         ...
>         self.dataset = load_dataset(
>             "HuggingFaceFW/fineweb-edu",
>             name=dataset_name,
>             split="train",
>             streaming=True
>         )
> ```
>
> Then for continued training, swap to a bigger split like `"sample-100BT"` or the full dataset.

---

## Part 5: Multi-Worker Sharding (Optional)

If you want to use `num_workers > 0` without seeing duplicate data, you need to shard the dataset across workers:

```python
def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    buffer = []

    dataset = self.dataset
    if worker_info is not None:
        # Shard: each worker skips to its portion
        # Worker 0 takes samples 0, 2, 4, ...
        # Worker 1 takes samples 1, 3, 5, ...
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        dataset = dataset.skip(worker_id)  # start at offset

        for i, doc in enumerate(dataset):
            if i % num_workers != 0:
                continue  # skip samples that belong to other workers

            tokens = self.tokenizer.encode(doc["text"]) + [self.tokenizer.eos_token_id]
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)
                yield x, y
    else:
        # Single worker — normal iteration
        for doc in dataset:
            tokens = self.tokenizer.encode(doc["text"]) + [self.tokenizer.eos_token_id]
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)
                yield x, y
```

---

## Part 6: Understanding the Loss

At the start of training, `loss` should be close to **`ln(vocab_size)`**:

```
ln(50257) ≈ 10.82
```

This makes sense — with random weights, the model assigns roughly equal probability to all 50,257 tokens, so `cross_entropy ≈ -log(1/50257) ≈ 10.82`.

As training progresses, loss should decrease:

```
Steps 0-100:       loss ≈ 10.8  (random predictions)
Steps 100-1000:    loss ≈ 8-9   (learning common tokens like "the", "is")
Steps 1000-10000:  loss ≈ 5-7   (learning word patterns)
Steps 10000+:      loss ≈ 3-4   (learning grammar, facts)
State of the art:  loss ≈ 2-3   (GPT-3 level on similar data)
```

> [!WARNING]
> If your initial loss is significantly higher than 10.82, there's likely a bug in your model or data pipeline. If loss doesn't decrease at all in the first 100 steps, check:
>
> 1. Is the learning rate too high or too low?
> 2. Are gradients flowing? (check `model.emb.weight.grad` isn't None after `loss.backward()`)
> 3. Is the data correct? (print a decoded `x` sample to verify it's real text)

---

## Quick Reference: Full File Structure

After implementing everything, your project should look like:

```
LLM/
├── model/
│   ├── __init__.py
│   ├── attn.py          ← GQFlashAttention, ScaledRoPE
│   ├── norm.py          ← RMSNorm
│   ├── transformer.py   ← Transformer block, SwiGLUFFN
│   └── llama.py         ← Model class (forward, prefill, decode, generate)
├── dataset.py           ← FineWebEduDataset (YOU CREATE THIS)
├── train.py             ← Training loop (YOU CREATE THIS)
├── test_generate.py     ← Inference test
├── inference_guide.md   ← How inference works
├── training_guide.md    ← This file
└── checkpoints/         ← Created during training
    ├── step_5000.pt
    ├── step_10000.pt
    └── ...
```

---

## Cheat Sheet: Key Decisions

| Decision        | Choice                              | Why                                                      |
| --------------- | ----------------------------------- | -------------------------------------------------------- |
| Dataset style   | `IterableDataset`                   | Streaming — can't use `__getitem__`                      |
| Token packing   | Concatenate + chunk                 | No padding waste, simple                                 |
| Optimizer       | AdamW                               | Standard for LLMs; decoupled weight decay                |
| β2              | 0.95                                | Better than 0.999 for pre-training (Llama/GPT-3 finding) |
| LR schedule     | Cosine + warmup                     | Smooth decay, avoids sharp drops                         |
| Mixed precision | bfloat16                            | 2x speed, no scaler needed                               |
| Gradient accum  | 8 steps                             | Large effective batch without OOM                        |
| Grad clipping   | 1.0                                 | Prevents training instability                            |
| Weight decay    | 0.1 (weights only)                  | Regularization; not applied to biases/norms              |
| Resumability    | Checkpoint model + optimizer + step | Exact continuation possible                              |
