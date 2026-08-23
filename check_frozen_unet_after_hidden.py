import torch


pretrained_path = (
    "runs/3k_pretrained_unet 2026.08.22--04-26-56/"
    "checkpoints/best.pyt"
)

hidden_path = (
    "runs/test_frozen_unet_hidden 2026.08.22--12-49-03/"
    "checkpoints/test_frozen_unet_hidden--epoch-2.pyt"
)


# ------------------------------------------------------------
# Load original pretrained U-Net
# ------------------------------------------------------------

pre = torch.load(
    pretrained_path,
    map_location="cpu",
    weights_only=False,
)

# adapt this list to the checkpoint format we used previously
for key in [
    "model_state_dict",
    "unet_state_dict",
    "state_dict",
    "model",
    "unet",
]:
    if isinstance(pre, dict) and key in pre:
        pre = pre[key]
        if hasattr(pre, "state_dict"):
            pre = pre.state_dict()
        break


# ------------------------------------------------------------
# Load U-Net stored inside HiDDeN checkpoint
# ------------------------------------------------------------

hidden = torch.load(
    hidden_path,
    map_location="cpu",
    weights_only=False,
)["enc-dec-model"]

prefix = "noiser.noise_layers.1.unet."

inside = {
    k[len(prefix):]: v
    for k, v in hidden.items()
    if k.startswith(prefix)
}


print("Original U-Net tensors:", len(pre))
print("HiDDeN U-Net tensors:", len(inside))
print()


all_equal = True

for key in pre:
    if key not in inside:
        print("MISSING:", key)
        all_equal = False
        continue

    diff = (pre[key] - inside[key]).abs()

    max_diff = diff.max().item()

    print(
        f"{key:60s} max_diff={max_diff:.12f}"
    )

    if max_diff != 0:
        all_equal = False


print()
print("=" * 70)

if all_equal:
    print("PASS: Frozen U-Net weights are EXACTLY unchanged.")
else:
    print("WARNING: U-Net weights changed or keys did not match.")

print("=" * 70)