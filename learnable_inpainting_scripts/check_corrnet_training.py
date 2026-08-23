import torch

p1 = (
    "runs/3k_learn_inpainting_union_mask 2026.05.25--19-10-20/"
    "checkpoints/3k_learn_inpainting_union_mask--epoch-1.pyt"
)

p400 = (
    "runs/3k_learn_inpainting_union_mask 2026.05.25--19-10-20/"
    "checkpoints/3k_learn_inpainting_union_mask--epoch-400.pyt"
)

a = torch.load(p1, map_location="cpu")["enc-dec-model"]
b = torch.load(p400, map_location="cpu")["enc-dec-model"]

keys = [k for k in a if "corrnet" in k.lower()]

print("Found CorrNet tensors:", len(keys))
print()

for k in keys:
    max_diff = (a[k] - b[k]).abs().max().item()
    mean_diff = (a[k] - b[k]).abs().mean().item()

    print(k)
    print(f"  max abs diff : {max_diff:.8f}")
    print(f"  mean abs diff: {mean_diff:.8f}")