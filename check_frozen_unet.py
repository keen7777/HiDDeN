import torch

from noise_layers.frozen_pretrained_unet_inpainting import (
    FrozenPretrainedUNetInpainting,
)


checkpoint_path = (
    "runs/3k_pretrained_unet 2026.08.22--04-26-56/"
    "checkpoints/best.pyt"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

layer = FrozenPretrainedUNetInpainting(
    checkpoint_path=checkpoint_path,
    min_ratio=0.1,
    max_ratio=0.4,
    seed=42,
).to(device)


print("=" * 70)
print("DEVICE")
print("=" * 70)
print(device)


# ----------------------------------------------------------
# 1. Check parameter freezing
# ----------------------------------------------------------
params = list(layer.unet.parameters())

total_params = sum(
    p.numel()
    for p in params
)

trainable_params = sum(
    p.numel()
    for p in params
    if p.requires_grad
)

print("\n" + "=" * 70)
print("PARAMETERS")
print("=" * 70)
print("Total U-Net parameters:", total_params)
print("Trainable U-Net parameters:", trainable_params)

assert trainable_params == 0


# ----------------------------------------------------------
# 2. Check gradient THROUGH frozen U-Net
# ----------------------------------------------------------
x = torch.randn(
    2,
    3,
    128,
    128,
    device=device,
    requires_grad=True,
)

cover = torch.randn_like(x)

output, _ = layer([x, cover])

loss = output.mean()
loss.backward()

print("\n" + "=" * 70)
print("GRADIENT CHECK")
print("=" * 70)

print(
    "Input gradient exists:",
    x.grad is not None,
)

print(
    "Input grad abs mean:",
    x.grad.abs().mean().item()
    if x.grad is not None
    else None,
)

unet_grad_count = sum(
    p.grad is not None
    for p in layer.unet.parameters()
)

print(
    "U-Net parameters with gradients:",
    unet_grad_count,
)

assert x.grad is not None
assert x.grad.abs().sum() > 0
assert unet_grad_count == 0


# ----------------------------------------------------------
# 3. Check mask coverage
# ----------------------------------------------------------
mask = layer.last_mask

coverage = mask.mean(
    dim=(1, 2, 3)
)

print("\n" + "=" * 70)
print("MASK")
print("=" * 70)

print("Mask shape:", tuple(mask.shape))
print("Coverage:", coverage.tolist())

assert torch.all(coverage >= 0.1 - 1e-3)
assert torch.all(coverage <= 0.4 + 1e-3)


# ----------------------------------------------------------
# 4. Known pixels must remain unchanged
# ----------------------------------------------------------
known = 1.0 - mask

max_known_diff = (
    (output - x).abs() * known
).max().item()

print("\n" + "=" * 70)
print("KNOWN PIXEL CHECK")
print("=" * 70)

print(
    "Max difference in known region:",
    max_known_diff,
)

assert max_known_diff < 1e-6


print("\n" + "=" * 70)
print("ALL CHECKS PASSED")
print("=" * 70)