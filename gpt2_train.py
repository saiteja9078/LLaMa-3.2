import os
import time
import json
import csv
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gpt2.model.gpt2 import GPT2
from train_utils.utils import *
from data.fineweb_dataset import FineWebDataset

# Suppress noisy tokenizer warnings about sequence length
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def train(config):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print("Using device:", device)
    print("Using dtype:", dtype)

    # Enable TF32 for faster matmuls on Ampere+ GPUs
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    model = GPT2(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_blocks=config["n_blocks"],
    ).to(device)

    # Separate weight-decay vs no-decay params (biases & layernorms don't decay)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        fused=(device == "cuda"),
    )

    step = 0
    tokens_seen = 0

    if config.get("resume_from"):
        if config.get("continue_mode"):
            step, tokens_seen = load_checkpoint_for_continued_training(
                config["resume_from"], model, optimizer, device
            )
        else:
            step, tokens_seen = load_checkpoint(
                config["resume_from"], model, optimizer, device
            )

    train_dataset = FineWebDataset(
        seq_len=config["max_seq_len"], # uses max_seq_len
        data_path=config.get("data_path"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=8,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
        drop_last=True,
    )

    data_iter = iter(train_loader)

    # If resuming, skip the tokens already seen
    if tokens_seen > 0:
        seq_to_skip = tokens_seen // config["max_seq_len"]
        print(f"Skipping {seq_to_skip} chunks to resume...")
        for i, _ in enumerate(data_iter):
            if i >= seq_to_skip:
                break

    model.train()
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # ── CSV logger (buffered writes – negligible overhead) ──
    os.makedirs(os.path.join("gpt2", "results"), exist_ok=True)
    csv_path = os.path.join("gpt2", "results", "training_log.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if csv_file.tell() == 0:          # write header only for new files
        csv_writer.writerow(["step", "loss", "lr", "tokens_seen", "tokens_per_sec", "step_time"])

    last_log_time = time.time()
    losses = []

    start_msg = f"Starting training from step {step} | target {config['max_steps']} steps"
    print(start_msg)
    with open(os.path.join("gpt2", "results", "training_output.log"), "a") as txt_log:
        txt_log.write(start_msg + "\n")

    while step < config["max_steps"]:

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(config["grad_accum_steps"]):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=dtype):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                loss = loss / config["grad_accum_steps"]

            loss.backward()
            loss_accum += loss.item()
            tokens_seen += x.size(0) * x.size(1)

        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        # Update learning rate
        lr = get_lr(
            step,
            config["warmup_steps"],
            config["max_steps"],
            config["learning_rate"],
            config["min_lr"],
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        step += 1
        losses.append(loss_accum)

        # Measure per-step throughput
        now = time.time()
        step_time = now - last_log_time
        last_log_time = now

        step_tokens = (
            config["batch_size"]
            * config["max_seq_len"]
            * config["grad_accum_steps"]
        )
        tokens_per_sec = step_tokens / step_time if step_time > 0 else 0

        log_str = (
            f"step {step:>6d} | "
            f"loss {loss_accum:.4f} | "
            f"lr {lr:.2e} | "
            f"tokens {tokens_seen:>12,} | "
            f"tok/s {tokens_per_sec:,.0f}"
        )
        print(log_str)
        with open(os.path.join("gpt2", "results", "training_output.log"), "a") as txt_log:
            txt_log.write(log_str + "\n")

        # Write row to CSV (flush every 100 steps to avoid I/O stalls)
        csv_writer.writerow([step, f"{loss_accum:.6f}", f"{lr:.2e}", tokens_seen,
                             f"{tokens_per_sec:.0f}", f"{step_time:.4f}"])
        if step % 100 == 0:
            csv_file.flush()

        if step % config["save_every"] == 0:
            save_checkpoint(
                model, optimizer, step, tokens_seen, config,
                os.path.join(config["checkpoint_dir"], f"step_{step}.pt"),
            )

    save_checkpoint(
        model, optimizer, step, tokens_seen, config,
        os.path.join(config["checkpoint_dir"], f"step_{step}_final.pt"),
    )

    csv_file.flush()
    csv_file.close()
    print(f"Training log saved to {csv_path}")
    end_msg = f"\nTraining complete! {tokens_seen:,} tokens processed in {step} steps."
    print(end_msg)
    with open(os.path.join("gpt2", "results", "training_output.log"), "a") as txt_log:
        txt_log.write(end_msg + "\n")
    return losses


if __name__ == "__main__":
    import argparse

    with open("gpt2/train_utils/config.json") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--continue_training", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=150_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    config["max_steps"] = args.max_steps
    config["batch_size"] = args.batch_size
    config["max_seq_len"] = args.seq_len # use max_seq_len for gpt2 config

    if args.data_path:
        config["data_path"] = args.data_path

    if args.resume:
        config["resume_from"] = args.resume
    elif args.continue_training:
        config["resume_from"] = args.continue_training
        config["continue_mode"] = True

    losses = train(config)

    os.makedirs(os.path.join("gpt2", "results"), exist_ok=True)
    with open(os.path.join("gpt2", "results", "losses.json"), "w") as f:
        json.dump({"loss": losses}, f)

    print("Losses saved to gpt2/results/losses.json")
