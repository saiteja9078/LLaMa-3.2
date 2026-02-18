# from model.llama import LLaMa
import torch
import torch.nn as nn
import json
from model.llama import LLaMa
from train_utils.utils import *
from data.fineweb_dataset import FineWebDataset
from torch.utils.data import DataLoader
import time

def train(config):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("Using device:",device,"\nUsing Dtype:",dtype)
    model = LLaMa(
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

    step,tokens_seen =0 ,0

    if config["resume_from"]:
        step,tokens_seen = load_checkpoint(
            config["resume_from"],model,optimizer,device
        )
    
    train_dataset = FineWebDataset(seq_len=config["seq_len"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],prefetch_factor=4 if device == "cuda" else None
                              )
    data_iter = iter(train_loader)

    #If resuming skip the tokens already seen
    if tokens_seen>0:
        seq_to_skip = tokens_seen // config["seq_len"]
        print(f"Skipping {seq_to_skip} chunks to resume:")
        for i,_ in enumerate(data_iter):
            if i>=seq_to_skip:
                break
    
    if device=="cuda" and hasattr(torch,"compile"):
        model = torch.compile(model)
        print("⚡ Model compiled with torch.compile")

    
    # Training Loop
    model.train()
    os.makedirs(config["checkpoint_dir"],exist_ok=True)
    t0 = time.time()
    losses = []
    while step < config["max_steps"]:
        optimizer.zero_grad()
        loss_acc = 0.

        #Gradient Accumulation
        for micro_step in range(config["grad_accum_steps"]):
            try:
                x,y = next(data_iter)
            except Exception:
                data_iter = iter(train_loader)
                x,y = next(data_iter)

            x,y =x.to(device),y.to(device)

            #Forward with mixed precision
            # Some ops → low precision (fast)
            # Sensitive ops → FP32 (stable)

            with torch.autocast(device_type=device,dtype=dtype):
                logits = model(x) #b ,t ,vocab
                loss = nn.functional.cross_entropy(
                    logits.view(-1,logits.size(-1)) #b*t, vocab
                    ,y.view(-1) #b*t
                )
                loss = loss / config["grad_accum_steps"]
            loss.backward()
            loss_accum = loss.item()
            losses.append(loss_accum)
            tokens_seen = x.numel()
        

        torch.nn.utils.clip_grad_norm_(model.parameters(),config["grad_clip"])

        #update lr
        lr = get_lr(step,config["warmup_steps"],config["max_steps"],config["learning_rate"],config["min_lr"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        optimizer.step()
        step+=1

        if step % 100 ==0 or step==1:
            dt = time.time() - t0
            tokens_per_sec = tokens_seen / dt if dt>0 else 0
            print(
                f"step {step:>6d} | "
                f"loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | "
                f"tokens {tokens_seen:>12,} | "
                f"tok/s {tokens_per_sec:,.0f}"
            )
        
        if step % config["save_every"] == 0:
            save_checkpoint(
                model,optimizer,step,tokens_seen,config,
                os.path.join(config["checkpoint_dir"], f"step_{step}_.pt")
            )
    save_checkpoint(
        model, optimizer, step, tokens_seen, config,
        os.path.join(config["checkpoint_dir"], f"step_{step}_final.pt")
    )
    print(f"\nTraining complete! {tokens_seen:,} tokens processed in {step} steps.")


if __name__ == "__main__":
    import argparse
    import json
    with open("train_utils/config.json") as f:
        config = json.load(f)
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
