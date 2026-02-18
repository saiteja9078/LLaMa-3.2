import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os

def get_lr(step,warmup_steps,max_steps,max_lr,min_lr):
    #Linear warmup
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    if step >= max_steps:
        return min_lr
    #cosine decay between warmup_steps and max_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 *(1 + math.cos(math.pi*progress))
    return min_lr + coeff *(max_lr - min_lr)

def save_checkpoint(model,optimizer,step,tokens_seen,config,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    checkpoint = {
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "step":step,
        "tokens_seen":tokens_seen,
        "config":config
    }
    torch.save(checkpoint,path)
    print(f"Checkpoint saved: {path} (step {step}, {tokens_seen:,} tokens)")

def load_checkpoint(path,model,optimizer,device):
    checkpoint = torch.load(path,map_location=device,weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    tokens_seen = checkpoint["tokens_seen"]
    step = checkpoint["step"]
    print(f"Resumed from {path} (step {step}, {tokens_seen:,} tokens)")
    return step, tokens_seen

# To continue training on NEW data with existing weights:
def load_checkpoint_for_continued_training(path, model, optimizer, device):
    """Load weights + optimizer state, but reset step counter."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    total_tokens_so_far = checkpoint["tokens_seen"]
    print(f"  Loaded weights from {path} ({total_tokens_so_far:,} tokens trained)")
    print(f"  Step reset to 0 — starting fresh LR schedule for continued training")
    return 0, total_tokens_so_far  # step=0, but keep total token count for logging
# Parameters
# total_steps = 1000
# warmup_steps = 100
# max_steps = 1000
# max_lr = 1e-3
# min_lr = 1e-5

# # Compute LR values
# steps = list(range(total_steps))
# lrs = [get_lr(step, warmup_steps, max_steps, max_lr, min_lr) for step in steps]

# # Plot (single chart, no specific colors)
# plt.figure()
# plt.plot(steps, lrs)
# plt.xlabel("Step")
# plt.ylabel("Learning Rate")
# plt.title("Warmup + Cosine Decay Learning Rate Schedule")
# plt.tight_layout()

# # Save file
# file_path = "results/lr_schedule.png"
# plt.savefig(file_path)
# plt.close()
