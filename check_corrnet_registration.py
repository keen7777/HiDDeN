import torch

checkpoints = {
    "old": "runs/3k_learn_inpainting 2026.05.25--05-19-37/checkpoints/3k_learn_inpainting--epoch-400.pyt",
    "union": "runs/3k_learn_inpainting_union_mask 2026.05.25--19-10-20/checkpoints/3k_learn_inpainting_union_mask--epoch-400.pyt",
}

for name, path in checkpoints.items():
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    ckpt = torch.load(path, map_location="cpu")

    state = ckpt["enc-dec-model"]

    corr_keys = [
        k for k in state.keys()
        if any(x in k.lower() for x in ["corrnet", "learnable", "inpainting"])
    ]

    print("Number of model tensors:", len(state))
    print("CorrNet-related keys:")
    for k in corr_keys:
        print("  ", k)

    if corr_keys:
        print("\n>>> CorrNet IS registered in this checkpoint.")
    else:
        print("\n>>> No registered CorrNet found in this checkpoint.")