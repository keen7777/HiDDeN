import os
import csv
import argparse

import numpy as np
import torch
import torchvision

import utils

from noise_layers.pretrained_cnn_inpainting import (
    ReconstructionCNN,
)

from noise_layers.pretrained_unet_inpainting import (
    SmallUNet,
    ControlledRectangleMaskGenerator,
)

from noise_layers.fill_strategies.mean_fill import MeanFill


# ============================================================
# Metrics
# ============================================================

def masked_l1_per_image(pred, target, mask):
    """
    pred / target:
        [B, 3, H, W]

    mask:
        [B, 1, H, W]

    Returns:
        [B]
    """

    error = (
        torch.abs(pred - target)
        * mask
    )

    numerator = (
        error
        .flatten(1)
        .sum(dim=1)
    )

    denominator = (
        mask
        .flatten(1)
        .sum(dim=1)
        * target.shape[1]
        + 1e-8
    )

    return (
        numerator
        / denominator
    )


def masked_mse_per_image(pred, target, mask):

    error = (
        (pred - target) ** 2
        * mask
    )

    numerator = (
        error
        .flatten(1)
        .sum(dim=1)
    )

    denominator = (
        mask
        .flatten(1)
        .sum(dim=1)
        * target.shape[1]
        + 1e-8
    )

    return (
        numerator
        / denominator
    )


def masked_psnr_per_image(pred, target, mask):
    """
    Images are in [-1, 1].

    Therefore:
        data_range = 2
    """

    mse = masked_mse_per_image(
        pred,
        target,
        mask,
    )

    data_range = torch.tensor(
        2.0,
        device=mse.device,
    )

    psnr = (
        20.0 * torch.log10(data_range)
        - 10.0 * torch.log10(
            mse + 1e-12
        )
    )

    return psnr


def full_psnr_per_image(pred, target):
    """
    Full-image PSNR.

    Since all methods preserve known pixels,
    this will naturally be much higher than
    masked-region PSNR.
    """

    mse = (
        (pred - target) ** 2
    ).flatten(1).mean(dim=1)

    data_range = torch.tensor(
        2.0,
        device=mse.device,
    )

    psnr = (
        20.0 * torch.log10(data_range)
        - 10.0 * torch.log10(
            mse + 1e-12
        )
    )

    return psnr


# ============================================================
# Mean Fill
# ============================================================

def apply_mean_fill(image, mask):
    """
    Apply existing MeanFill implementation.

    image:
        [B,3,H,W]

    mask:
        [B,1,H,W]

    MeanFill expects one image + [H,W] mask.
    """

    strategy = MeanFill()

    outputs = []

    for b in range(
        image.shape[0]
    ):

        output = strategy.fill(
            image[b:b + 1],
            mask[b, 0],
        )

        outputs.append(
            output
        )

    return torch.cat(
        outputs,
        dim=0,
    )


# ============================================================
# Visualization
# ============================================================

def save_examples(
    original,
    mask,
    zero_fill,
    mean_fill,
    shallow_cnn,
    unet,
    output_file,
    n=8,
):
    """
    Save six rows:

        1. original
        2. masked input
        3. zero fill
        4. mean fill
        5. shallow CNN
        6. U-Net
    """

    n = min(
        n,
        original.shape[0],
    )

    original = (
        original[:n]
        .detach()
        .cpu()
    )

    mask = (
        mask[:n]
        .detach()
        .cpu()
    )

    zero_fill = (
        zero_fill[:n]
        .detach()
        .cpu()
    )

    mean_fill = (
        mean_fill[:n]
        .detach()
        .cpu()
    )

    shallow_cnn = (
        shallow_cnn[:n]
        .detach()
        .cpu()
    )

    unet = (
        unet[:n]
        .detach()
        .cpu()
    )

    masked = (
        original
        * (1.0 - mask)
    )

    def vis(x):

        return torch.clamp(
            (x + 1.0) / 2.0,
            0.0,
            1.0,
        )

    grid = torch.cat(
        [
            vis(original),
            vis(masked),
            vis(zero_fill),
            vis(mean_fill),
            vis(shallow_cnn),
            vis(unet),
        ],
        dim=0,
    )

    torchvision.utils.save_image(
        grid,
        output_file,
        nrow=n,
    )


# ============================================================
# Checkpoint loading
# ============================================================

def load_shallow_cnn(
    checkpoint_path,
    device,
):

    model = ReconstructionCNN(
        hidden_channels=32
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model" in checkpoint:

        state_dict = checkpoint[
            "model"
        ]

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        raise KeyError(
            "Could not find shallow CNN "
            "state_dict in checkpoint."
        )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model, checkpoint


def load_unet(
    checkpoint_path,
    device,
):

    model = SmallUNet(
        base_channels=32
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model" in checkpoint:

        state_dict = checkpoint[
            "model"
        ]

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        raise KeyError(
            "Could not find U-Net "
            "state_dict in checkpoint."
        )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model, checkpoint


# ============================================================
# Evaluate one fixed mask ratio
# ============================================================

def evaluate_one_ratio(
    shallow_model,
    unet_model,
    val_loader,
    device,
    ratio,
    mask_seed,
    output_dir,
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Controlled coverage.
    #
    # min=max and randomize_ratio=False means every image
    # receives the requested ratio.
    # --------------------------------------------------------

    mask_generator = (
        ControlledRectangleMaskGenerator(
            min_ratio=ratio,
            max_ratio=ratio,
            min_mask_size=8,
            max_aspect_ratio=4.0,
            seed=mask_seed,
            randomize_ratio=False,
        )
    )

    methods = [
        "ZeroFill",
        "MeanFill",
        "ShallowCNN",
        "UNet",
    ]

    metrics = {
        name: {
            "l1": [],
            "masked_psnr": [],
            "full_psnr": [],
        }
        for name in methods
    }

    actual_coverages = []

    examples_saved = False

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for image, _ in val_loader:

            image = image.to(
                device=device,
                dtype=torch.float32,
            )

            B, C, H, W = (
                image.shape
            )

            # =================================================
            # ONE mask only.
            #
            # All methods receive exactly this same mask.
            # =================================================

            mask = (
                mask_generator.generate(
                    B,
                    H,
                    W,
                    device,
                )
            )

            coverage = mask.mean(
                dim=(1, 2, 3)
            )

            actual_coverages.extend(
                coverage
                .cpu()
                .tolist()
            )

            # =================================================
            # 1. Zero Fill
            # =================================================

            zero_output = (
                image
                * (1.0 - mask)
            )

            # =================================================
            # 2. Mean Fill
            # =================================================

            mean_output = (
                apply_mean_fill(
                    image,
                    mask,
                )
            )

            # =================================================
            # 3. Pretrained shallow CNN
            # =================================================

            shallow_output = (
                shallow_model(
                    image,
                    mask,
                )
            )

            # =================================================
            # 4. Pretrained U-Net
            # =================================================

            unet_output = (
                unet_model(
                    image,
                    mask,
                )
            )

            outputs = {
                "ZeroFill":
                    zero_output,

                "MeanFill":
                    mean_output,

                "ShallowCNN":
                    shallow_output,

                "UNet":
                    unet_output,
            }

            # =================================================
            # Metrics
            # =================================================

            for (
                name,
                output
            ) in outputs.items():

                l1 = (
                    masked_l1_per_image(
                        output,
                        image,
                        mask,
                    )
                )

                masked_psnr = (
                    masked_psnr_per_image(
                        output,
                        image,
                        mask,
                    )
                )

                full_psnr = (
                    full_psnr_per_image(
                        output,
                        image,
                    )
                )

                metrics[
                    name
                ]["l1"].extend(
                    l1
                    .cpu()
                    .tolist()
                )

                metrics[
                    name
                ]["masked_psnr"].extend(
                    masked_psnr
                    .cpu()
                    .tolist()
                )

                metrics[
                    name
                ]["full_psnr"].extend(
                    full_psnr
                    .cpu()
                    .tolist()
                )

            # =================================================
            # Visual example
            # =================================================

            if not examples_saved:

                save_examples(
                    original=image,
                    mask=mask,

                    zero_fill=
                        zero_output,

                    mean_fill=
                        mean_output,

                    shallow_cnn=
                        shallow_output,

                    unet=
                        unet_output,

                    output_file=os.path.join(
                        output_dir,
                        (
                            f"comparison_"
                            f"ratio_{ratio:.3f}.png"
                        ),
                    ),
                )

                examples_saved = True

    # ========================================================
    # Aggregate
    # ========================================================

    results = {}

    for name in methods:

        results[name] = {
            "masked_l1":
                float(
                    np.mean(
                        metrics[
                            name
                        ]["l1"]
                    )
                ),

            "masked_psnr":
                float(
                    np.mean(
                        metrics[
                            name
                        ]["masked_psnr"]
                    )
                ),

            "full_psnr":
                float(
                    np.mean(
                        metrics[
                            name
                        ]["full_psnr"]
                    )
                ),
        }

    results["_coverage"] = {
        "target":
            float(ratio),

        "mean":
            float(
                np.mean(
                    actual_coverages
                )
            ),

        "std":
            float(
                np.std(
                    actual_coverages
                )
            ),

        "min":
            float(
                np.min(
                    actual_coverages
                )
            ),

        "max":
            float(
                np.max(
                    actual_coverages
                )
            ),
    }

    return results


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--options-file",
        required=True,
        type=str,
        help=(
            "Path to HiDDeN "
            "options-and-config.pickle."
        ),
    )

    parser.add_argument(
        "--shallow-checkpoint",
        required=True,
        type=str,
        help=(
            "Checkpoint of pretrained "
            "shallow reconstruction CNN."
        ),
    )

    parser.add_argument(
        "--unet-checkpoint",
        required=True,
        type=str,
        help=(
            "Checkpoint of pretrained "
            "U-Net reconstruction model."
        ),
    )

    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ],
    )

    parser.add_argument(
        "--mask-seed",
        default=1042,
        type=int,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "reconstruction_quality_sweep"
        ),
        type=str,
    )

    args = parser.parse_args()

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "Reconstruction quality sweep"
    )
    print("=" * 80)

    print(
        "Device:",
        device
    )

    print(
        "Ratios:",
        args.ratios
    )

    print(
        "Mask seed:",
        args.mask_seed
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # ========================================================
    # Validation data
    # ========================================================

    (
        train_options,
        hidden_config,
        _
    ) = utils.load_options(
        args.options_file
    )

    _, val_loader = (
        utils.get_data_loaders(
            hidden_config,
            train_options,
        )
    )

    print(
        "Validation images:",
        len(val_loader.dataset)
    )

    # ========================================================
    # Models
    # ========================================================

    (
        shallow_model,
        shallow_checkpoint,
    ) = load_shallow_cnn(
        args.shallow_checkpoint,
        device,
    )

    (
        unet_model,
        unet_checkpoint,
    ) = load_unet(
        args.unet_checkpoint,
        device,
    )

    print()
    print(
        "Shallow CNN checkpoint epoch:",
        shallow_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "Shallow CNN val loss:",
        shallow_checkpoint.get(
            "val_loss",
            "unknown",
        ),
    )

    print()

    print(
        "U-Net checkpoint epoch:",
        unet_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "U-Net val loss:",
        unet_checkpoint.get(
            "val_loss",
            "unknown",
        ),
    )

    print(
        "U-Net val L1:",
        unet_checkpoint.get(
            "val_l1",
            "unknown",
        ),
    )

    print(
        "U-Net val MSE:",
        unet_checkpoint.get(
            "val_mse",
            "unknown",
        ),
    )

    # ========================================================
    # CSV
    # ========================================================

    csv_file = os.path.join(
        args.output_dir,
        "quality_sweep.csv",
    )

    with open(
        csv_file,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "target_ratio",
            "actual_coverage_mean",
            "actual_coverage_std",
            "actual_coverage_min",
            "actual_coverage_max",
            "method",
            "masked_l1",
            "masked_psnr",
            "full_psnr",
        ])

    # ========================================================
    # Sweep
    # ========================================================

    all_results = {}

    for ratio in args.ratios:

        print()
        print("=" * 80)

        print(
            f"Mask ratio = "
            f"{ratio:.3f}"
        )

        print("=" * 80)

        results = (
            evaluate_one_ratio(
                shallow_model=
                    shallow_model,

                unet_model=
                    unet_model,

                val_loader=
                    val_loader,

                device=
                    device,

                ratio=
                    ratio,

                mask_seed=
                    args.mask_seed,

                output_dir=
                    args.output_dir,
            )
        )

        all_results[
            ratio
        ] = results

        coverage = (
            results[
                "_coverage"
            ]
        )

        print()

        print(
            "Actual coverage:"
        )

        print(
            f"  mean = "
            f"{coverage['mean']:.6f}"
        )

        print(
            f"  std  = "
            f"{coverage['std']:.6f}"
        )

        print(
            f"  min  = "
            f"{coverage['min']:.6f}"
        )

        print(
            f"  max  = "
            f"{coverage['max']:.6f}"
        )

        # ----------------------------------------------------
        # Method output
        # ----------------------------------------------------

        for method in [
            "ZeroFill",
            "MeanFill",
            "ShallowCNN",
            "UNet",
        ]:

            values = (
                results[
                    method
                ]
            )

            print()
            print(method)

            print(
                f"  Masked L1   = "
                f"{values['masked_l1']:.6f}"
            )

            print(
                f"  Masked PSNR = "
                f"{values['masked_psnr']:.4f} dB"
            )

            print(
                f"  Full PSNR   = "
                f"{values['full_psnr']:.4f} dB"
            )

            with open(
                csv_file,
                "a",
                newline="",
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    ratio,
                    coverage["mean"],
                    coverage["std"],
                    coverage["min"],
                    coverage["max"],
                    method,
                    values["masked_l1"],
                    values["masked_psnr"],
                    values["full_psnr"],
                ])

    # ========================================================
    # Coverage summary
    # ========================================================

    print()
    print("=" * 100)

    print(
        "SUMMARY: TARGET VS ACTUAL COVERAGE"
    )

    print("=" * 100)

    print(
        f"{'target':>10}"
        f"{'actual':>12}"
        f"{'std':>12}"
        f"{'min':>12}"
        f"{'max':>12}"
    )

    for ratio in args.ratios:

        c = (
            all_results[
                ratio
            ]["_coverage"]
        )

        print(
            f"{ratio:>10.3f}"
            f"{c['mean']:>12.4f}"
            f"{c['std']:>12.4f}"
            f"{c['min']:>12.4f}"
            f"{c['max']:>12.4f}"
        )

    # ========================================================
    # L1 summary
    # ========================================================

    print()
    print("=" * 100)

    print(
        "SUMMARY: MASKED L1"
    )

    print("=" * 100)

    print(
        f"{'ratio':>8}"
        f"{'actual':>10}"
        f"{'Zero':>12}"
        f"{'Mean':>12}"
        f"{'Shallow':>12}"
        f"{'UNet':>12}"
        f"{'UNet-Mean':>14}"
    )

    for ratio in args.ratios:

        r = all_results[
            ratio
        ]

        actual = (
            r[
                "_coverage"
            ]["mean"]
        )

        zero = (
            r[
                "ZeroFill"
            ]["masked_l1"]
        )

        mean = (
            r[
                "MeanFill"
            ]["masked_l1"]
        )

        shallow = (
            r[
                "ShallowCNN"
            ]["masked_l1"]
        )

        unet = (
            r[
                "UNet"
            ]["masked_l1"]
        )

        print(
            f"{ratio:>8.3f}"
            f"{actual:>10.4f}"
            f"{zero:>12.4f}"
            f"{mean:>12.4f}"
            f"{shallow:>12.4f}"
            f"{unet:>12.4f}"
            f"{unet - mean:>+14.4f}"
        )

    # ========================================================
    # PSNR summary
    # ========================================================

    print()
    print("=" * 100)

    print(
        "SUMMARY: MASKED PSNR"
    )

    print("=" * 100)

    print(
        f"{'ratio':>8}"
        f"{'actual':>10}"
        f"{'Zero':>12}"
        f"{'Mean':>12}"
        f"{'Shallow':>12}"
        f"{'UNet':>12}"
        f"{'UNet-Mean':>14}"
    )

    for ratio in args.ratios:

        r = all_results[
            ratio
        ]

        actual = (
            r[
                "_coverage"
            ]["mean"]
        )

        zero = (
            r[
                "ZeroFill"
            ]["masked_psnr"]
        )

        mean = (
            r[
                "MeanFill"
            ]["masked_psnr"]
        )

        shallow = (
            r[
                "ShallowCNN"
            ]["masked_psnr"]
        )

        unet = (
            r[
                "UNet"
            ]["masked_psnr"]
        )

        print(
            f"{ratio:>8.3f}"
            f"{actual:>10.4f}"
            f"{zero:>12.2f}"
            f"{mean:>12.2f}"
            f"{shallow:>12.2f}"
            f"{unet:>12.2f}"
            f"{unet - mean:>+14.2f}"
        )

    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 80)

    print(
        "CSV saved to:"
    )

    print(
        csv_file
    )

    print()

    print(
        "Visual comparisons saved to:"
    )

    print(
        args.output_dir
    )

    print("=" * 80)


if __name__ == "__main__":
    main()