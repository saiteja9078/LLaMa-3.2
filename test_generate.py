import torch
from transformers import AutoTokenizer
from model.llama import Model

# ── GPT-2 BPE tokenizer ──
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ── Small model for testing (random weights, won't produce coherent text) ──
pe_config = {
    "d": 64,  # head_dim = d_model // n_q_heads = 512 // 8 = 64
    "base": 500_000.0,
    "scaling_factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

model = Model(
    pe_config=pe_config,
    vocab_size=tokenizer.vocab_size,  # 50257
    d_model=512,
    n_q_heads=8,
    n_kv_heads=2,
    n_blocks=4,
    hidden_dim=1024,
)
model.eval()

# ── Tokenize prompt ──
prompt = "The capital of France is"
token_ids = tokenizer.encode(prompt, return_tensors="pt")  # (1, T)
print(f"Prompt: {prompt}")
print(f"Token IDs: {token_ids.tolist()}")
print(f"Generating...\n")

# ── Stream generated tokens ──
print(prompt, flush=True)
for token_text in model.generate(
    token_ids,
    tokenizer,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
):
    print(token_text, end="", flush=True)

print("\n\nDone!")
