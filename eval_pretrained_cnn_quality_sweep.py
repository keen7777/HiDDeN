import os
import csv
import argparse

import numpy as np
import torch

import utils

from noise_layers.pretrained_cnn_inpainting import (
    ReconstructionCNN,
    RectangleMaskGenerator,
)

from eval_pretrained_cnn_quality import (
    masked_l1_per_image,
    masked_psnr_per_image,
    full_psnr_per_image,
    apply_mean_fill,
    save_examples,
)


# ============================================================
# Evaluate one fixed target mask ratio
# ============================================================

def evaluate_one_ratio(
    model,
    val_loader,
    device,
    ratio,
    mask_seed,
    output_dir,
):
    """
    Evaluate all reconstruction methods using exactly the same
    validation images and masks at one fixed target mask ratio.

    Important:
        target ratio != necessarily exact realized ratio

    because the union of rectangles may overshoot the target
    when the final rectangle is added.

    Therefore actual pixel-level mask coverage is recorded.
    """

    # --------------------------------------------------------
    # Fixed target ratio
    # --------------------------------------------------------

    mask_generator = RectangleMaskGenerator(
        min_ratio=ratio,
        max_ratio=ratio,
        seed=mask_seed,
    )

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

    # Store actual realized mask coverage per image.
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

            B, C, H, W = image.shape

            # ------------------------------------------------
            # Generate mask ONCE.
            #
            # All three methods receive exactly the same mask.
            # ------------------------------------------------

            mask = mask_generator.generate(
                B,
                H,
                W,
                device,
            )

            # ------------------------------------------------
            # Record actual pixel-level mask coverage.
            #
            # mask shape:
            # [B, 1, H, W]
            #
            # mean over (1,2,3) gives one coverage value
            # for each image in the batch.
            # ------------------------------------------------

            batch_coverages = mask.mean(
                dim=(1, 2, 3)
            )

            actual_coverages.extend(
                batch_coverages.cpu().tolist()
            )

            # =================================================
            # 1. Zero Fill
            # =================================================
            #
            # Images are normalized to [-1,1].
            #
            # Setting masked region to 0 therefore corresponds
            # to middle grey in image space.
            # =================================================

            zero_output = (
                image * (1.0 - mask)
            )

            # =================================================
            # 2. Mean Fill
            # =================================================

            mean_output = apply_mean_fill(
                image,
                mask,
            )

            # =================================================
            # 3. Pretrained shallow CNN
            # =================================================

            cnn_output = model(
                image,
                mask,
            )

            outputs = {
                "ZeroFill": zero_output,
                "MeanFill": mean_output,
                "PretrainedCNN": cnn_output,
            }

            # =================================================
            # Metrics
            # =================================================

            for name, output in outputs.items():

                l1 = masked_l1_per_image(
                    output,
                    image,
                    mask,
                )

                masked_psnr = masked_psnr_per_image(
                    output,
                    image,
                    mask,
                )

                full_psnr = full_psnr_per_image(
                    output,
                    image,
                )

                metrics[name]["l1"].extend(
                    l1.cpu().tolist()
                )

                metrics[name]["masked_psnr"].extend(
                    masked_psnr.cpu().tolist()
                )

                metrics[name]["full_psnr"].extend(
                    full_psnr.cpu().tolist()
                )

            # =================================================
            # Save first batch for visual inspection
            # =================================================

            if not examples_saved:

                save_examples(
                    original=image,
                    mask=mask,
                    zero_fill=zero_output,
                    mean_fill=mean_output,
                    cnn_fill=cnn_output,
                    output_file=os.path.join(
                        output_dir,
                        f"comparison_target_{ratio:.3f}.png",
                    ),
                )

                examples_saved = True

    # ========================================================
    # Aggregate metrics
    # ========================================================

    results = {}

    for name in methods:

        results[name] = {
            "masked_l1":
                float(
                    np.mean(
                        metrics[name]["l1"]
                    )
                ),

            "masked_psnr":
                float(
                    np.mean(
                        metrics[name]["masked_psnr"]
                    )
                ),

            "full_psnr":
                float(
                    np.mean(
                        metrics[name]["full_psnr"]
                    )
                ),
        }

    # --------------------------------------------------------
    # Actual coverage statistics
    # --------------------------------------------------------

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
            "Path to a HiDDeN "
            "options-and-config.pickle file."
        ),
    )

    parser.add_argument(
        "--cnn-checkpoint",
        required=True,
        type=str,
        help=(
            "Path to pretrained CNN "
            "checkpoint, e.g. best.pyt."
        ),
    )

    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[
            0.025,
            0.05,
            0.075,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
        ],
        help=(
            "Target mask ratios to sweep."
        ),
    )

    parser.add_argument(
        "--mask-seed",
        type=int,
        default=1042,
        help=(
            "Seed for deterministic "
            "mask generation."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "pretrained_cnn_quality_sweep"
        ),
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

    print("=" * 70)
    print(
        "CNN reconstruction quality sweep"
    )
    print("=" * 70)

    print(
        "Device:",
        device
    )

    print(
        "Target ratios:",
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
    # Validation loader
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
    # Load pretrained CNN
    # ========================================================

    model = ReconstructionCNN(
        hidden_channels=32
    ).to(device)

    checkpoint = torch.load(
        args.cnn_checkpoint,
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
            "Could not find CNN "
            "state_dict in checkpoint."
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
        "CNN checkpoint val loss:",
        checkpoint.get(
            "val_loss",
            "unknown"
        )
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
        print("=" * 70)

        print(
            f"Target mask ratio = "
            f"{ratio:.3f}"
        )

        print("=" * 70)

        results = evaluate_one_ratio(
            model=model,
            val_loader=val_loader,
            device=device,
            ratio=ratio,
            mask_seed=args.mask_seed,
            output_dir=args.output_dir,
        )

        all_results[ratio] = results

        coverage = results[
            "_coverage"
        ]

        # ----------------------------------------------------
        # Coverage information
        # ----------------------------------------------------

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

        print(
            f"  overshoot(mean) = "
            f"{coverage['mean'] - ratio:+.6f}"
        )

        # ----------------------------------------------------
        # Method metrics
        # ----------------------------------------------------

        for method in [
            "ZeroFill",
            "MeanFill",
            "PretrainedCNN",
        ]:

            values = results[
                method
            ]

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

            # ----------------------------------------------
            # Save to CSV
            # ----------------------------------------------

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
    # Compact coverage summary
    # ========================================================

    print()
    print("=" * 92)
    print(
        "SUMMARY: TARGET VS ACTUAL COVERAGE"
    )
    print("=" * 92)

    print(
        f"{'target':>10}"
        f"{'mean':>12}"
        f"{'std':>12}"
        f"{'min':>12}"
        f"{'max':>12}"
        f"{'overshoot':>14}"
    )

    for ratio in args.ratios:

        c = all_results[
            ratio
        ]["_coverage"]

        print(
            f"{ratio:>10.3f}"
            f"{c['mean']:>12.4f}"
            f"{c['std']:>12.4f}"
            f"{c['min']:>12.4f}"
            f"{c['max']:>12.4f}"
            f"{c['mean'] - ratio:>+14.4f}"
        )

    # ========================================================
    # Masked L1 summary
    # ========================================================

    print()
    print("=" * 92)
    print(
        "SUMMARY: MASKED L1"
    )
    print("=" * 92)

    print(
        f"{'target':>10}"
        f"{'actual':>12}"
        f"{'ZeroFill':>14}"
        f"{'MeanFill':>14}"
        f"{'CNN':>14}"
        f"{'CNN-Mean':>14}"
    )

    for ratio in args.ratios:

        r = all_results[
            ratio
        ]

        actual = r[
            "_coverage"
        ]["mean"]

        zero = r[
            "ZeroFill"
        ]["masked_l1"]

        mean = r[
            "MeanFill"
        ]["masked_l1"]

        cnn = r[
            "PretrainedCNN"
        ]["masked_l1"]

        print(
            f"{ratio:>10.3f}"
            f"{actual:>12.4f}"
            f"{zero:>14.4f}"
            f"{mean:>14.4f}"
            f"{cnn:>14.4f}"
            f"{cnn - mean:>+14.4f}"
        )

    # ========================================================
    # Masked PSNR summary
    # ========================================================

    print()
    print("=" * 92)
    print(
        "SUMMARY: MASKED PSNR"
    )
    print("=" * 92)

    print(
        f"{'target':>10}"
        f"{'actual':>12}"
        f"{'ZeroFill':>14}"
        f"{'MeanFill':>14}"
        f"{'CNN':>14}"
        f"{'CNN-Mean':>14}"
    )

    for ratio in args.ratios:

        r = all_results[
            ratio
        ]

        actual = r[
            "_coverage"
        ]["mean"]

        zero = r[
            "ZeroFill"
        ]["masked_psnr"]

        mean = r[
            "MeanFill"
        ]["masked_psnr"]

        cnn = r[
            "PretrainedCNN"
        ]["masked_psnr"]

        print(
            f"{ratio:>10.3f}"
            f"{actual:>12.4f}"
            f"{zero:>14.2f}"
            f"{mean:>14.2f}"
            f"{cnn:>14.2f}"
            f"{cnn - mean:>+14.2f}"
        )

    # ========================================================
    # Final information
    # ========================================================

    print()
    print("=" * 70)

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

    print("=" * 70)


if __name__ == "__main__":
    main()