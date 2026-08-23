import os
import json
import random
import argparse

import numpy as np
import torch
import torchvision.utils as vutils

import utils

from model.hidden import Hidden
from average_meter import AverageMeter

from evaluation.rgb_yuv_convert import convert_img_range
from evaluation.psnr_ssim_evaluation import (
    compute_psnr,
    compute_ssim,
)

from noise_layers.identity import Identity
from noise_layers.eval_inpainting import EvalInpainting

from noise_layers.fill_strategies.mean_fill import MeanFill
from noise_layers.fill_strategies.telea_fill import TeleaFill
from noise_layers.fill_strategies.navier_stokes_fill import NavierStokesFill
from noise_layers.fill_strategies.shiftmap_fill import ShiftMapFill


# ============================================================
# EXTERNAL EVALUATION ATTACKS
# ============================================================
#
# Inpainting attacks all use the SAME random rectangle
# removal-mask generator implemented inside EvalInpainting.
#
# Mask convention:
#
#   removal_mask = 1 -> pixel is removed / reconstructed
#   removal_mask = 0 -> pixel is retained / known
#
# Identity is used separately for clean evaluation:
#
#   encoded image -> Identity -> decoder
#
# ============================================================

FILL_MAP = {
    "mean": MeanFill,
    "telea": TeleaFill,
    "navier": NavierStokesFill,
    "shiftmap": ShiftMapFill,
}


AVAILABLE_ATTACKS = [
    "identity",
    "mean",
    "telea",
    "navier",
    "shiftmap",
]


# ============================================================
# BUILD EXTERNAL ATTACK
# ============================================================

def build_external_attack(
    name,
    removal_ratio,
    seed,
):
    """
    Build one external evaluation attack.

    Parameters
    ----------
    name : str
        One of:
            identity
            mean
            telea
            navier
            shiftmap

    removal_ratio : float
        Fraction of image pixels removed before reconstruction.

        Only relevant for inpainting attacks.

        Example:
            0.1 -> approximately 10% removed
            0.9 -> approximately 90% removed

    seed : int
        Seed used for deterministic mask generation.

    Returns
    -------
    nn.Module
        External evaluation distortion.
    """

    # ========================================================
    # CLEAN / IDENTITY EVALUATION
    # ========================================================

    if name == "identity":
        return Identity()

    # ========================================================
    # INPAINTING ATTACKS
    # ========================================================

    if name not in FILL_MAP:
        raise ValueError(
            f"Unknown attack '{name}'. "
            f"Available attacks: {AVAILABLE_ATTACKS}"
        )

    fill_strategy = FILL_MAP[name]()

    attack = EvalInpainting(
        max_mask_ratio=removal_ratio,

        # Random rectangle-mask parameters.
        max_mask_number=100,
        min_mask_size=8,
        max_aspect_ratio=3.0,

        fill_strategy=fill_strategy,

        # Sweep uses exactly the requested removal ratio
        # rather than randomly sampling one.
        randomize_ratio=False,

        # Same seed:
        # same model/image/ratio -> same mask sequence.
        seed=seed,
    )

    return attack


# ============================================================
# LOAD MODEL FOR EXTERNAL EVALUATION
# ============================================================

def load_model_for_external_eval(
    hidden_config,
    checkpoint,
    device,
    external_noiser,
):
    """
    Load a trained HiDDeN model while replacing its original
    training noiser with an external evaluation attack.

    Important
    ---------
    We restore:
        - trained encoder
        - trained decoder
        - trained discriminator

    We deliberately DO NOT restore:
        - the original training noiser
        - optimizer state

    This is necessary for runs such as the frozen pretrained
    U-Net model, where the training noiser itself contains
    registered parameters.

    External evaluation should instead use the externally
    supplied attack:
        Identity
        MeanFill
        Telea
        Navier-Stokes
        ShiftMap
    """

    model = Hidden(
        hidden_config,
        device,
        external_noiser,
        tb_logger=None,
    )

    # --------------------------------------------------------
    # Encoder / decoder checkpoint
    # --------------------------------------------------------

    checkpoint_state = checkpoint[
        "enc-dec-model"
    ]

    # Remove all parameters belonging to the TRAINING noiser.
    filtered_state = {
        key: value
        for key, value in checkpoint_state.items()
        if not key.startswith("noiser.")
    }

    load_result = (
        model.encoder_decoder.load_state_dict(
            filtered_state,
            strict=False,
        )
    )

    # --------------------------------------------------------
    # Verify checkpoint compatibility
    # --------------------------------------------------------
    #
    # Missing noiser keys are expected because we deliberately
    # replace the training noiser.
    #
    # Any other missing/unexpected keys are suspicious.
    # --------------------------------------------------------

    bad_missing = [
        key
        for key in load_result.missing_keys
        if not key.startswith("noiser.")
    ]

    bad_unexpected = [
        key
        for key in load_result.unexpected_keys
        if not key.startswith("noiser.")
    ]

    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Unexpected checkpoint mismatch.\n"
            f"Missing keys: {bad_missing}\n"
            f"Unexpected keys: {bad_unexpected}"
        )

    # --------------------------------------------------------
    # Restore discriminator
    # --------------------------------------------------------

    if "discrim-model" in checkpoint:
        model.discriminator.load_state_dict(
            checkpoint["discrim-model"]
        )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.encoder_decoder.eval()

    if hasattr(model, "discriminator"):
        model.discriminator.eval()

    return model


# ============================================================
# HELPER
# ============================================================

def to_float(value):
    """
    Convert a PyTorch scalar or numeric value to Python float.
    """

    if torch.is_tensor(value):
        return (
            value
            .detach()
            .cpu()
            .item()
        )

    return float(value)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # DEVICE
    # ========================================================
    #
    # Classical reconstruction methods use OpenCV on CPU.
    # Keeping the complete evaluation on CPU also simplifies
    # reproducibility.
    # ========================================================

    device = torch.device("cpu")

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "External robustness evaluation "
            "for trained HiDDeN models."
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "-d",
        "--data-dir",
        required=True,
        type=str,
        help=(
            "Dataset root containing the val folder."
        ),
    )

    # --------------------------------------------------------
    # Trained model
    # --------------------------------------------------------

    parser.add_argument(
        "-r",
        "--runs-root",
        "--runs_root",
        dest="runs_root",
        default="./runs",
        type=str,
        help=(
            "Root directory containing trained runs."
        ),
    )

    parser.add_argument(
        "--run-name",
        required=True,
        type=str,
        help=(
            "Exact trained run folder name."
        ),
    )

    # --------------------------------------------------------
    # External attack
    # --------------------------------------------------------

    parser.add_argument(
        "--attack",
        required=True,
        type=str,
        choices=AVAILABLE_ATTACKS,
        help=(
            "External evaluation attack."
        ),
    )

    # --------------------------------------------------------
    # Removal ratio sweep
    # --------------------------------------------------------

    parser.add_argument(
        "--min-ratio",
        "--min",
        dest="min_ratio",
        type=float,
        default=0.1,
        help=(
            "Minimum fraction of pixels removed."
        ),
    )

    parser.add_argument(
        "--max-ratio",
        "--max",
        dest="max_ratio",
        type=float,
        default=0.9,
        help=(
            "Maximum fraction of pixels removed."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help=(
            "Number of removal ratios in the sweep."
        ),
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed used for messages, masks, "
            "and evaluation randomness."
        ),
    )

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help=(
            "Evaluation batch size. "
            "Batch size 1 is recommended "
            "for final evaluation."
        ),
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default="sweep_results",
        help=(
            "Directory used to save JSON results "
            "and debug images."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Save example cover / encoded / attacked "
            "/ mask images."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # ARGUMENT VALIDATION
    # ========================================================

    if not (
        0.0
        <= args.min_ratio
        <= args.max_ratio
        <= 1.0
    ):
        raise ValueError(
            "Require "
            "0 <= min_ratio <= max_ratio <= 1."
        )

    if args.steps < 1:
        raise ValueError(
            "--steps must be >= 1."
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # ========================================================
    # LOAD RUN CONFIGURATION
    # ========================================================

    run_path = os.path.join(
        args.runs_root,
        args.run_name,
    )

    options_file = os.path.join(
        run_path,
        "options-and-config.pickle",
    )

    (
        train_options,
        hidden_config,
        noise_config,
    ) = utils.load_options(
        options_file
    )

    # Evaluation uses the validation set only.
    train_options.validation_folder = (
        os.path.join(
            args.data_dir,
            "val",
        )
    )

    train_options.batch_size = (
        args.batch_size
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint, checkpoint_file = (
        utils.load_last_checkpoint(
            os.path.join(
                run_path,
                "checkpoints",
            )
        )
    )

    # ========================================================
    # PRINT CONFIGURATION
    # ========================================================

    print("=" * 60)
    print("EXTERNAL ROBUSTNESS EVALUATION")
    print("=" * 60)

    print(
        f"Run: "
        f"{args.run_name}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_file}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Attack: "
        f"{args.attack}"
    )

    print(
        f"Seed: "
        f"{args.seed}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    print("=" * 60)

    # ========================================================
    # VALIDATION LOADER
    # ========================================================

    _, val_data = (
        utils.get_data_loaders(
            hidden_config,
            train_options,
        )
    )

    file_count = len(
        val_data.dataset
    )

    print(
        f"Validation images: "
        f"{file_count}"
    )

    # ========================================================
    # FIXED WATERMARK MESSAGES
    # ========================================================
    #
    # Generate ALL validation messages exactly once.
    #
    # Therefore validation image i receives exactly the same
    # watermark message:
    #
    #   across removal ratios
    #   across reconstruction methods
    #   across trained models
    #   during identity evaluation
    #
    # The message RNG is independent of attack RNG.
    # ========================================================

    message_generator = (
        torch.Generator(
            device="cpu"
        )
    )

    message_generator.manual_seed(
        args.seed
    )

    fixed_messages = torch.randint(
        low=0,
        high=2,
        size=(
            file_count,
            hidden_config.message_length,
        ),
        generator=message_generator,
        dtype=torch.float32,
    )

    print(
        f"Fixed messages generated "
        f"for {file_count} images."
    )

    # ========================================================
    # REMOVAL RATIOS
    # ========================================================
    #
    # Identity:
    #   Only run once at ratio 0.
    #
    # Inpainting:
    #   Run the requested sweep.
    # ========================================================

    if args.attack == "identity":

        removal_ratios = [
            0.0
        ]

    else:

        removal_ratios = np.linspace(
            args.min_ratio,
            args.max_ratio,
            args.steps,
        )

    # ========================================================
    # MODEL NAME
    # ========================================================

    model_name = (
        args.run_name.split(" ")[0]
    )

    # ========================================================
    # RESULT METADATA
    # ========================================================

    if args.attack == "identity":

        ratio_definition = (
            "Identity evaluation: "
            "no pixels are removed."
        )

    else:

        ratio_definition = (
            "removal_ratio = fraction of image pixels "
            "removed before reconstruction"
        )

    results = {
        "run_name": args.run_name,

        "model_name": model_name,

        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),

        "attack": args.attack,

        "seed": args.seed,

        "num_images": file_count,

        "batch_size": args.batch_size,

        "mask_convention": (
            "1 = removed/reconstructed, "
            "0 = retained/known"
        ),

        "ratio_definition": (
            ratio_definition
        ),

        "results": {},
    }

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    for removal_ratio in removal_ratios:

        removal_ratio = float(
            removal_ratio
        )

        print()
        print("=" * 60)

        if args.attack == "identity":

            print(
                "IDENTITY / CLEAN EVALUATION"
            )

        else:

            print(
                f"{args.attack.upper()} "
                f"| removal ratio = "
                f"{removal_ratio:.3f}"
            )

        print("=" * 60)

        # ====================================================
        # RESET RANDOM STATES
        # ====================================================
        #
        # EvalInpainting has its own seeded RNG.
        # Resetting global RNGs additionally protects against
        # stochastic behaviour in future attack implementations.
        # ====================================================

        random.seed(
            args.seed
        )

        np.random.seed(
            args.seed
        )

        torch.manual_seed(
            args.seed
        )

        # ====================================================
        # BUILD EXTERNAL ATTACK
        # ====================================================

        external_attack = (
            build_external_attack(
                name=args.attack,
                removal_ratio=removal_ratio,
                seed=args.seed,
            )
        )

        # ====================================================
        # LOAD WATERMARK MODEL
        #
        # The original training noiser is deliberately NOT
        # restored.
        # ====================================================

        model = (
            load_model_for_external_eval(
                hidden_config=hidden_config,
                checkpoint=checkpoint,
                device=device,
                external_noiser=external_attack,
            )
        )

        # ====================================================
        # METRIC METERS
        # ====================================================

        psnr_meter = (
            AverageMeter()
        )

        ssim_meter = (
            AverageMeter()
        )

        ber_meter = (
            AverageMeter()
        )

        actual_ratio_meter = (
            AverageMeter()
        )

        message_offset = 0

        debug_saved = False

        # ====================================================
        # DATASET EVALUATION
        # ====================================================

        with torch.no_grad():

            for image, _ in val_data:

                image = image.to(
                    device=device,
                    dtype=torch.float32,
                )

                current_batch_size = (
                    image.shape[0]
                )

                # ============================================
                # FIXED MESSAGE
                # ============================================

                message = fixed_messages[
                    message_offset:
                    message_offset
                    + current_batch_size
                ].to(device)

                message_offset += (
                    current_batch_size
                )

                # ============================================
                # HiDDeN FORWARD / VALIDATION
                # ============================================

                losses, (
                    encoded_images,
                    noised_images,
                    decoded_messages,
                ) = model.validate_on_batch(
                    [
                        image,
                        message,
                    ]
                )

                # ============================================
                # CONVERT RANGE
                #
                # [-1, 1] -> [0, 1]
                # ============================================

                cover = convert_img_range(
                    image
                )

                encoded = convert_img_range(
                    encoded_images
                )

                attacked = convert_img_range(
                    noised_images
                )

                cover = torch.clamp(
                    cover,
                    0.0,
                    1.0,
                )

                encoded = torch.clamp(
                    encoded,
                    0.0,
                    1.0,
                )

                attacked = torch.clamp(
                    attacked,
                    0.0,
                    1.0,
                )

                # ============================================
                # IMAGE QUALITY
                # ============================================
                #
                # For inpainting:
                #
                #   cover vs attacked / reconstructed
                #
                # For identity:
                #
                #   attacked == encoded
                #
                # therefore this automatically measures:
                #
                #   cover vs encoded
                #
                # i.e. embedding quality.
                # ============================================

                psnr = compute_psnr(
                    cover,
                    attacked,
                )

                ssim = compute_ssim(
                    cover,
                    attacked,
                )

                psnr_meter.update(
                    psnr.item(),
                    current_batch_size,
                )

                ssim_meter.update(
                    ssim.item(),
                    current_batch_size,
                )

                # ============================================
                # WATERMARK BER
                # ============================================

                ber_value = to_float(
                    losses[
                        "bitwise-error  "
                    ]
                )

                ber_meter.update(
                    ber_value,
                    current_batch_size,
                )

                # ============================================
                # ACTUAL REMOVAL RATIO
                # ============================================
                #
                # Only EvalInpainting exposes last_masks.
                # Identity has no removal mask.
                # ============================================

                if (
                    args.attack != "identity"
                    and hasattr(
                        external_attack,
                        "last_masks",
                    )
                    and external_attack.last_masks
                    is not None
                ):

                    for removal_mask in (
                        external_attack.last_masks
                    ):

                        actual_ratio = float(
                            removal_mask
                            .float()
                            .mean()
                            .item()
                        )

                        actual_ratio_meter.update(
                            actual_ratio
                        )

                # ============================================
                # DEBUG OUTPUT
                #
                # Save only first image for each ratio.
                # ============================================

                if (
                    args.debug
                    and not debug_saved
                ):

                    debug_dir = os.path.join(
                        args.output_dir,
                        "debug",
                        (
                            f"M_{model_name}"
                            f"_A_{args.attack}"
                        ),
                    )

                    os.makedirs(
                        debug_dir,
                        exist_ok=True,
                    )

                    ratio_tag = (
                        f"{removal_ratio:.3f}"
                    )

                    # ----------------------------------------
                    # Cover
                    # ----------------------------------------

                    vutils.save_image(
                        cover[0],
                        os.path.join(
                            debug_dir,
                            (
                                f"cover_"
                                f"{ratio_tag}.png"
                            ),
                        ),
                    )

                    # ----------------------------------------
                    # Encoded / watermarked
                    # ----------------------------------------

                    vutils.save_image(
                        encoded[0],
                        os.path.join(
                            debug_dir,
                            (
                                f"encoded_"
                                f"{ratio_tag}.png"
                            ),
                        ),
                    )

                    # ----------------------------------------
                    # Attacked
                    #
                    # For identity this should equal encoded.
                    # ----------------------------------------

                    vutils.save_image(
                        attacked[0],
                        os.path.join(
                            debug_dir,
                            (
                                f"attacked_"
                                f"{ratio_tag}.png"
                            ),
                        ),
                    )

                    # ----------------------------------------
                    # Removal mask
                    #
                    # Only available for inpainting attacks.
                    # ----------------------------------------

                    if (
                        args.attack != "identity"
                        and hasattr(
                            external_attack,
                            "last_masks",
                        )
                        and external_attack.last_masks
                    ):

                        removal_mask = (
                            external_attack
                            .last_masks[0]
                            .float()
                            .cpu()
                        )

                        vutils.save_image(
                            removal_mask.unsqueeze(0),
                            os.path.join(
                                debug_dir,
                                (
                                    f"removal_mask_"
                                    f"{ratio_tag}.png"
                                ),
                            ),
                        )

                        # ----------------------------------------
                        # Masked image
                        #
                        # Visualizes the encoded image after
                        # removing the pixels specified by the
                        # removal mask, before reconstruction.
                        #
                        # removal_mask = 1 -> removed
                        # removal_mask = 0 -> retained
                        # ----------------------------------------

                        encoded_cpu = (
                            encoded[0]
                            .detach()
                            .cpu()
                        )

                        masked = (
                            encoded_cpu
                            * (1.0 - removal_mask.unsqueeze(0))
                        )

                        vutils.save_image(
                            masked,
                            os.path.join(
                                debug_dir,
                                (
                                    f"masked_"
                                    f"{ratio_tag}.png"
                                ),
                            ),
                        )

                    debug_saved = True

        # ====================================================
        # DATASET-LEVEL ACTUAL REMOVAL RATIO
        # ====================================================

        if args.attack == "identity":

            mean_actual_ratio = 0.0

        elif actual_ratio_meter.count > 0:

            mean_actual_ratio = float(
                actual_ratio_meter.avg
            )

        else:

            mean_actual_ratio = None

        # ====================================================
        # SAVE RESULT FOR THIS RATIO
        # ====================================================

        ratio_key = (
            f"{removal_ratio:.3f}"
        )

        results["results"][
            ratio_key
        ] = {
            "target_removal_ratio": (
                removal_ratio
            ),

            "actual_removal_ratio": (
                mean_actual_ratio
            ),

            "psnr": float(
                psnr_meter.avg
            ),

            "ssim": float(
                ssim_meter.avg
            ),

            "ber": float(
                ber_meter.avg
            ),
        }

        # ====================================================
        # PRINT RESULT
        # ====================================================

        if args.attack == "identity":

            print(
                "Target removal ratio: "
                "0.000000"
            )

            print(
                "Actual removal ratio: "
                "0.000000"
            )

        else:

            print(
                f"Target removal ratio: "
                f"{removal_ratio:.6f}"
            )

            if (
                mean_actual_ratio
                is not None
            ):

                print(
                    f"Actual removal ratio: "
                    f"{mean_actual_ratio:.6f}"
                )

        print(
            f"PSNR: "
            f"{psnr_meter.avg:.4f}"
        )

        print(
            f"SSIM: "
            f"{ssim_meter.avg:.4f}"
        )

        print(
            f"BER: "
            f"{ber_meter.avg:.6f}"
        )

    # ========================================================
    # OUTPUT FILE NAME
    # ========================================================

    if args.attack == "identity":

        output_filename = (
            f"clean_"
            f"M_{model_name}_"
            f"S_{args.seed}.json"
        )

    else:

        output_filename = (
            f"sweep_"
            f"M_{model_name}_"
            f"A_{args.attack}_"
            f"R_{args.min_ratio:.2f}_"
            f"{args.max_ratio:.2f}_"
            f"S_{args.seed}.json"
        )

    output_file = os.path.join(
        args.output_dir,
        output_filename,
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)

    print(
        f"Saved -> "
        f"{output_file}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()