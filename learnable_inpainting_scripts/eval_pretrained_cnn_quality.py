import os
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import utils

from noise_layers.pretrained_cnn_inpainting import (
    ReconstructionCNN,
    RectangleMaskGenerator,
)

from noise_layers.fill_strategies.mean_fill import MeanFill


# ============================================================
# Metrics
# ============================================================

def masked_l1_per_image(pred, target, mask):
    """
    pred/target: [B,3,H,W]
    mask:        [B,1,H,W]

    Returns:
        [B]
    """

    error = torch.abs(pred - target) * mask

    numerator = error.flatten(1).sum(dim=1)

    denominator = (
        mask.flatten(1).sum(dim=1)
        * target.shape[1]
        + 1e-8
    )

    return numerator / denominator


def masked_mse_per_image(pred, target, mask):

    error = (
        (pred - target) ** 2
    ) * mask

    numerator = error.flatten(1).sum(dim=1)

    denominator = (
        mask.flatten(1).sum(dim=1)
        * target.shape[1]
        + 1e-8
    )

    return numerator / denominator


def masked_psnr_per_image(pred, target, mask):
    """
    Images are in [-1,1], therefore data_range = 2.
    """

    mse = masked_mse_per_image(
        pred,
        target,
        mask
    )

    return (
        20.0 * torch.log10(
            torch.tensor(
                2.0,
                device=mse.device
            )
        )
        - 10.0 * torch.log10(
            mse + 1e-12
        )
    )


def full_psnr_per_image(pred, target):
    """
    Full-image PSNR.

    Note:
    Known pixels are unchanged, so this will naturally
    be higher than masked-region PSNR.
    """

    mse = (
        (pred - target) ** 2
    ).flatten(1).mean(dim=1)

    return (
        20.0 * torch.log10(
            torch.tensor(
                2.0,
                device=mse.device
            )
        )
        - 10.0 * torch.log10(
            mse + 1e-12
        )
    )


# ============================================================
# Mean fill
# ============================================================

def apply_mean_fill(image, mask):
    """
    Apply existing MeanFill to a batch.

    MeanFill expects:
        image: [1,C,H,W]
        mask:  [H,W]
    """

    strategy = MeanFill()

    outputs = []

    for b in range(image.shape[0]):

        filled = strategy.fill(
            image[b:b+1],
            mask[b, 0]
        )

        outputs.append(filled)

    return torch.cat(
        outputs,
        dim=0
    )


# ============================================================
# Save visual comparison
# ============================================================

def save_examples(
    original,
    mask,
    zero_fill,
    mean_fill,
    cnn_fill,
    output_file,
    n=8,
):

    n = min(
        n,
        original.shape[0]
    )

    original = original[:n].cpu()
    mask = mask[:n].cpu()

    zero_fill = zero_fill[:n].cpu()
    mean_fill = mean_fill[:n].cpu()
    cnn_fill = cnn_fill[:n].cpu()

    # Pure masked input
    masked = (
        original
        * (1.0 - mask)
    )

    # [-1,1] -> [0,1]
    def vis(x):
        return torch.clamp(
            (x + 1.0) / 2.0,
            0.0,
            1.0
        )

    grid = torch.cat(
        [
            vis(original),   # row 1
            vis(masked),     # row 2
            vis(zero_fill),  # row 3
            vis(mean_fill),  # row 4
            vis(cnn_fill),   # row 5
        ],
        dim=0
    )

    torchvision.utils.save_image(
        grid,
        output_file,
        nrow=n
    )


# ============================================================
# Main evaluation
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--options-file",
        required=True,
        type=str,
        help="HiDDeN options-and-config.pickle"
    )

    parser.add_argument(
        "--cnn-checkpoint",
        required=True,
        type=str,
        help="Path to pretrained CNN best.pyt"
    )

    parser.add_argument(
        "--mask-seed",
        default=1042,
        type=int,
        help=(
            "Use 1042 to match the validation "
            "masks used during CNN pretraining."
        )
    )

    parser.add_argument(
        "--min-ratio",
        default=0.1,
        type=float
    )

    parser.add_argument(
        "--max-ratio",
        default=0.5,
        type=float
    )

    parser.add_argument(
        "--output-dir",
        default="pretrained_cnn_quality_eval",
        type=str
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("CNN reconstruction quality evaluation")
    print("=" * 70)

    print("Device:", device)

    # ========================================================
    # Load original validation setup
    # ========================================================

    (
        train_options,
        hidden_config,
        _
    ) = utils.load_options(
        args.options_file
    )

    _, val_loader = utils.get_data_loaders(
        hidden_config,
        train_options
    )

    print(
        "Validation images:",
        len(val_loader.dataset)
    )

    print(
        "Mask ratio:",
        args.min_ratio,
        "-",
        args.max_ratio
    )

    print(
        "Mask seed:",
        args.mask_seed
    )

    # ========================================================
    # Load CNN
    # ========================================================

    model = ReconstructionCNN(
        hidden_channels=32
    ).to(device)

    checkpoint = torch.load(
        args.cnn_checkpoint,
        map_location=device
    )

    # Compatible with both versions of the save script
    if "model" in checkpoint:
        state_dict = checkpoint["model"]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    else:
        raise KeyError(
            "Could not find CNN model state "
            "inside checkpoint."
        )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "CNN checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    print(
        "CNN validation loss:",
        checkpoint.get(
            "val_loss",
            "unknown"
        )
    )

    # ========================================================
    # ONE deterministic mask stream
    # ========================================================

    mask_generator = RectangleMaskGenerator(
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        seed=args.mask_seed
    )

    # ========================================================
    # Accumulators
    # ========================================================

    methods = [
        "ZeroFill",
        "MeanFill",
        "PretrainedCNN",
    ]

    metrics = {
        name: {
            "l1": [],
            "masked_psnr": [],
            "full_psnr": [],
        }
        for name in methods
    }

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    examples_saved = False

    # ========================================================
    # Evaluation loop
    # ========================================================

    with torch.no_grad():

        for image, _ in val_loader:

            image = image.to(
                device=device,
                dtype=torch.float32
            )

            B, C, H, W = image.shape

            # IMPORTANT:
            # Generate the mask ONCE.
            #
            # All methods receive this exact same mask.
            mask = mask_generator.generate(
                B,
                H,
                W,
                device
            )

            # ------------------------------------------------
            # 1. ZeroFill
            #
            # In normalized [-1,1] space:
            # zero = middle gray.
            # ------------------------------------------------

            zero_output = (
                image
                * (1.0 - mask)
            )

            # ------------------------------------------------
            # 2. MeanFill
            # ------------------------------------------------

            mean_output = apply_mean_fill(
                image,
                mask
            )

            # ------------------------------------------------
            # 3. Pretrained shallow CNN
            # ------------------------------------------------

            cnn_output = model(
                image,
                mask
            )

            outputs = {
                "ZeroFill":
                    zero_output,

                "MeanFill":
                    mean_output,

                "PretrainedCNN":
                    cnn_output,
            }

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            for name, output in outputs.items():

                l1 = masked_l1_per_image(
                    output,
                    image,
                    mask
                )

                masked_psnr = (
                    masked_psnr_per_image(
                        output,
                        image,
                        mask
                    )
                )

                full_psnr = (
                    full_psnr_per_image(
                        output,
                        image
                    )
                )

                metrics[name]["l1"].extend(
                    l1.cpu().tolist()
                )

                metrics[name][
                    "masked_psnr"
                ].extend(
                    masked_psnr.cpu().tolist()
                )

                metrics[name][
                    "full_psnr"
                ].extend(
                    full_psnr.cpu().tolist()
                )

            # ------------------------------------------------
            # Visual examples
            # ------------------------------------------------

            if not examples_saved:

                save_examples(
                    original=image,
                    mask=mask,
                    zero_fill=zero_output,
                    mean_fill=mean_output,
                    cnn_fill=cnn_output,
                    output_file=os.path.join(
                        args.output_dir,
                        "comparison.png"
                    )
                )

                examples_saved = True

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name in methods:

        l1 = np.mean(
            metrics[name]["l1"]
        )

        masked_psnr = np.mean(
            metrics[name]["masked_psnr"]
        )

        full_psnr = np.mean(
            metrics[name]["full_psnr"]
        )

        print()
        print(name)
        print("-" * 40)

        print(
            f"Masked L1   = "
            f"{l1:.6f}"
        )

        print(
            f"Masked PSNR = "
            f"{masked_psnr:.4f} dB"
        )

        print(
            f"Full PSNR   = "
            f"{full_psnr:.4f} dB"
        )

    print()
    print(
        "Visual comparison saved to:",
        os.path.join(
            args.output_dir,
            "comparison.png"
        )
    )


if __name__ == "__main__":
    main()