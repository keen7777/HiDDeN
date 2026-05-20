import torch

ckpt = torch.load("experiments/crop-0.2-0.25/checkpoints/epoch-300.pyt", map_location="cpu")
print(type(ckpt))
print(ckpt.keys())