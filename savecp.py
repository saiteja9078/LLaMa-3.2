# import torch
# import sys
# import os

# path = r'C:\Users\Bajavada-stu\Desktop\LLaMa-3.2\checkpoints\step_10000.pt'

# print(f"Loading {path} ...")
# ckpt = torch.load(path, map_location="cpu")

# print("Original keys:", ckpt.keys())

# # Create lightweight inference checkpoint
# inference_ckpt = {
#     "model_state_dict": ckpt["model_state_dict"],
#     "config": ckpt["config"]
# }

# new_path = path.replace(".pt", "_inference.pt")

# torch.save(inference_ckpt, new_path)

# print(f"\nSaved inference checkpoint to {new_path}")

# orig_size = os.path.getsize(path) / (1024**3)
# new_size = os.path.getsize(new_path) / (1024**3)

# print(f"Original size: {orig_size:.2f} GB")
# print(f"Inference size: {new_size:.2f} GB")

import torch

src = "checkpoints/step_10000_inference.pt"
dst = "checkpoints/step_10000_inference_fp16.pt"

ckpt = torch.load(src, map_location="cpu")

ckpt["model_state_dict"] = {
    k: v.half()
    for k, v in ckpt["model_state_dict"].items()
}

torch.save(ckpt, dst)

print("Saved:", dst)