import torch
from gpt2.model.gpt2 import GPT2
import json

def test():
    print("Loading config...")
    with open('gpt2/train_utils/config.json', 'r') as f:
        config = json.load(f)
    
    print("Instantiating GPT-2...")
    model = GPT2(
        vocab_size=config['vocab_size'],
        max_seq_len=config['max_seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_blocks=config['n_blocks']
    )
    
    # Test text size
    batch_size = 2
    seq_len = 10
    
    print("Testing forward pass...")
    dummy_input = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    logits = model(dummy_input)
    assert logits.shape == (batch_size, seq_len, config['vocab_size'])
    print(f"Logits shape: {logits.shape} OK!")
    
    print("Testing prefill and decode (KV caching)...")
    n_layers = len(model.transformer_blocks)
    kv_caches = [{} for _ in range(n_layers)]
    
    prefill_logits = model.prefill(dummy_input, kv_caches)
    assert prefill_logits.shape == (batch_size, seq_len, config['vocab_size'])
    print(f"Prefill logits shape: {prefill_logits.shape} OK!")
    
    new_token = torch.randint(0, config['vocab_size'], (batch_size, 1))
    decode_logits, new_caches = model.decode(new_token, kv_caches)
    assert decode_logits.shape == (batch_size, 1, config['vocab_size'])
    print(f"Decode logits shape: {decode_logits.shape} OK!")
    
    print("All tests passed!")

if __name__ == "__main__":
    test()
