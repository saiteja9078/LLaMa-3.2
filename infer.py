"""
infer.py — Load a trained LLaMa checkpoint and generate text interactively.

Usage:
    python infer.py --checkpoint checkpoints/step_10000_inference.pt
    python infer.py --checkpoint checkpoints/step_10000_inference.pt --prompt "Once upon a time"
    python infer.py --checkpoint checkpoints/step_10000_inference.pt --max_tokens 200 --temperature 0.7 --top_k 40
    python infer.py --checkpoint checkpoints/step_10000_inference.pt --interactive
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.llama import LLaMa


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """
    Load a LLaMa model from a training checkpoint.
    The checkpoint contains the full config, so no external config file is needed.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    step = checkpoint.get("step", "?")
    tokens_seen = checkpoint.get("tokens_seen", "?")

    print(f"  Step: {step}")
    print(f"  Tokens seen: {tokens_seen:,}" if isinstance(tokens_seen, int) else f"  Tokens seen: {tokens_seen}")

    # Reconstruct the model from the saved config
    model = LLaMa(
        pe_config=config["pe_config"],
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_q_heads=config["n_q_heads"],
        n_kv_heads=config["n_kv_heads"],
        n_blocks=config["n_blocks"],
        hidden_dim=config["hidden_dim"],
    )

    # Handle torch.compile'd state dicts (keys may have _orig_mod. prefix)
    state_dict = checkpoint["model_state_dict"]
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        # Strip "_orig_mod." prefix added by torch.compile
        new_key = k.replace("_orig_mod.", "")
        cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()

    print(f"  Model loaded on {device} with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, config


@torch.no_grad()
def generate_text(
    model: LLaMa,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu",
):
    """
    Generate text from a prompt with proper UTF-8 handling.

    GPT-2 uses byte-level BPE, so multi-byte characters (smart quotes,
    em-dashes, etc.) can be split across multiple tokens. Decoding tokens
    one-at-a-time produces broken UTF-8 (the '��' garbage). Instead, we
    accumulate all generated token IDs and re-decode the full sequence
    each step, printing only the new characters.
    """
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)

    n_layers = len(model.transformer_blocks)
    kv_caches = [{} for _ in range(n_layers)]

    # Prefill: process the full prompt
    logits = model.prefill(input_ids, kv_caches)
    next_logits = logits[:, -1, :] / temperature

    eos_id = getattr(tokenizer, 'eos_token_id', None) or getattr(tokenizer, 'eos_id', None)

    generated_ids = []
    prev_text = ""

    for _ in range(max_new_tokens):
        # Top-k filtering
        if top_k > 0:
            top_vals, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < top_vals[:, -1:]] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        token_id = next_token.item()

        # Check EOS
        if eos_id is not None and token_id == eos_id:
            break

        generated_ids.append(token_id)

        # Decode ALL generated tokens together → proper multi-byte UTF-8
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Print only the new characters since last decode
        new_chars = full_text[len(prev_text):]
        if new_chars:
            print(new_chars, end="", flush=True)
        prev_text = full_text

        # Decode step: feed only the new token, grow the KV cache
        logits, kv_caches = model.decode(next_token, kv_caches)
        next_logits = logits[:, -1, :] / temperature

    print()  # newline after generation
    return prompt + prev_text


def main():
    parser = argparse.ArgumentParser(description="Inference with a trained LLaMa model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the checkpoint .pt file (e.g. checkpoints/step_10000_inference.pt)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt to generate from. If not provided, enters interactive mode.",
    )
    parser.add_argument("--max_tokens", type=int, default=150, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode for multiple prompts")
    parser.add_argument("--device", type=str, default=None, help="Device to use (auto-detected if not set)")
    parser.add_argument("--fp16", action="store_true", help="Cast model to float16 for faster inference")
    args = parser.parse_args()

    # ── Device selection ──
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    print()

    # ── Load model ──
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    if args.fp16:
        model = model.half()
        print("  Model cast to float16")

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"  Tokenizer: GPT-2 (vocab_size={tokenizer.vocab_size})")
    print()

    # ── Generation ──
    gen_kwargs = dict(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    if args.interactive or args.prompt is None:
        # Interactive loop
        print("=" * 60)
        print("  INTERACTIVE MODE  —  type 'quit' or 'exit' to stop")
        print("=" * 60)
        print()
        while True:
            try:
                prompt = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            print()
            generate_text(model, tokenizer, prompt, **gen_kwargs)
            print()
    else:
        # Single prompt
        print(f"Prompt: {args.prompt}")
        print("-" * 60)
        generate_text(model, tokenizer, args.prompt, **gen_kwargs)
        print("-" * 60)


if __name__ == "__main__":
    main()
