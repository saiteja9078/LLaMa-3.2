import argparse
import torch
import json
from transformers import AutoTokenizer
from gpt2.model.gpt2 import GPT2
from train_utils.utils import load_checkpoint

def load_gpt2_checkpoint(checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint["config"]
    step = checkpoint.get("step", "?")
    tokens_seen = checkpoint.get("tokens_seen", "?")
    
    print(f"  Step: {step}")
    print(f"  Tokens seen: {tokens_seen}")
    
    model = GPT2(
        vocab_size=config.get("vocab_size", 50257),
        max_seq_len=config.get("max_seq_len", 1024),
        d_model=config.get("d_model", 768),
        n_heads=config.get("n_heads", 12),
        n_blocks=config.get("n_blocks", 12),
    ).to(device)
    
    # Handle potentially compiled model names
    state_dict = checkpoint["model_state_dict"]
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_state_dict[k.replace("_orig_mod.", "")] = v
        
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    
    return model, config

class TiktokenWrapper:
    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.eos_token_id = self.enc.eot_token
        self.vocab_size = self.enc.n_vocab
    
    def encode(self, text, return_tensors=None):
        ids = self.enc.encode(text)
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return self.enc.decode(ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to GPT2 checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode for multiple prompts")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model, config = load_gpt2_checkpoint(args.checkpoint, device)
    tokenizer = TiktokenWrapper()
    
    if args.interactive:
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
                
            print("-" * 50)
            print(prompt, end="", flush=True)
            token_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            for next_text in model.generate(
                token_ids, 
                tokenizer, 
                max_new_tokens=args.max_tokens, 
                temperature=args.temperature, 
                top_k=args.top_k
            ):
                print(next_text, end="", flush=True)
            print("\n" + "-" * 50)
            print()
    else:
        print(f"\nPrompt: {args.prompt}")
        print("-" * 50)
        print(args.prompt, end="", flush=True)
        
        token_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        
        for next_text in model.generate(
            token_ids, 
            tokenizer, 
            max_new_tokens=args.max_tokens, 
            temperature=args.temperature, 
            top_k=args.top_k
        ):
            print(next_text, end="", flush=True)
            
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
