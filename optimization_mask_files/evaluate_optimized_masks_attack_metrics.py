import os
import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils

import utils
from model.hidden import Hidden
from average_meter import AverageMeter
from evaluation.rgb_yuv_convert import convert_img_range
from evaluation.psnr_ssim_evaluation import compute_psnr, compute_ssim

from noise_layers.fill_strategies.mean_fill import MeanFill
from noise_layers.fill_strategies.telea_fill import TeleaFill
from noise_layers.fill_strategies.navier_stokes_fill import NavierStokesFill
from noise_layers.fill_strategies.diffusion_fill import DiffusionFill


DEFAULT_MASK_FILE = (
    "/home/keen/HiDDeN/evaluation_data/"
    "opt_masks_diff_150_nlpe_100_den_0.1/"
    "val_ps_nlpe_masks_density_100.npy"
)

FILL_MAP = {
    "mean": MeanFill,
    "telea": TeleaFill,
    "navier": NavierStokesFill,
    "diffusion": DiffusionFill,
}


def build_fill_strategy(name, tau=0.25, diffusion_iterations=150):
    if name not in FILL_MAP:
        raise ValueError(
            f"Unknown attack '{name}'. Available attacks: {list(FILL_MAP.keys())}"
        )

    if name == "diffusion":
        return DiffusionFill(
            tau=tau,
            num_iterations=diffusion_iterations,
        )

    return FILL_MAP[name]()


class PrecomputedOptimizedMaskInpainting(nn.Module):
    """
    Apply one fill strategy using precomputed optimized sparse masks.

    Stored .npy mask convention:
        1 / True  = retained / known
        0 / False = removed / reconstructed

    Fill-strategy convention:
        1 = removed / reconstructed
        0 = retained / known
    """

    def __init__(self, mask_file, fill_strategy):
        super().__init__()

        self.mask_file = mask_file
        self.masks = np.load(mask_file)

        if self.masks.ndim != 3:
            raise ValueError(
                "Expected mask array with shape [N, H, W], "
                f"got {self.masks.shape}"
            )

        self.masks = self.masks.astype(np.bool_, copy=False)
        self.fill_strategy = fill_strategy
        self.next_mask_index = 0
        self.last_masks = None

    def reset(self):
        self.next_mask_index = 0
        self.last_masks = None

    def get_mask_batch(self, batch_size):
        start = self.next_mask_index
        end = start + batch_size

        if end > len(self.masks):
            raise IndexError(
                "Not enough precomputed masks. "
                f"Requested [{start}:{end}], but only {len(self.masks)} exist."
            )

        return self.masks[start:end], start, end

    def forward(self, encoded_and_cover):
        if not isinstance(encoded_and_cover, (list, tuple)):
            raise TypeError("Expected [encoded_images, cover_images].")

        if len(encoded_and_cover) != 2:
            raise ValueError("Expected exactly [encoded_images, cover_images].")

        encoded_images, cover_images = encoded_and_cover
        batch_size = encoded_images.size(0)
        batch_masks, _, end = self.get_mask_batch(batch_size)

        attacked_images = []
        used_removal_masks = []

        for batch_index in range(batch_size):
            retention_np = batch_masks[batch_index]

            if retention_np.shape != tuple(encoded_images.shape[-2:]):
                raise ValueError(
                    "Mask/image shape mismatch: "
                    f"mask={retention_np.shape}, "
                    f"image={tuple(encoded_images.shape[-2:])}"
                )

            # IMPORTANT: optimized masks store 1=retained,
            # while every FillStrategy expects 1=removed.
            removal_np = ~retention_np

            removal_mask = torch.from_numpy(
                removal_np.astype(np.float32)
            ).to(
                device=encoded_images.device,
                dtype=encoded_images.dtype,
            )

            encoded_single = encoded_images[
                batch_index : batch_index + 1
            ]

            reconstructed = self.fill_strategy.fill(
                encoded_single,
                removal_mask,
            )

            if reconstructed.shape != encoded_single.shape:
                raise ValueError(
                    "Fill strategy returned unexpected shape: "
                    f"{tuple(reconstructed.shape)}, "
                    f"expected {tuple(encoded_single.shape)}"
                )

            # Enforce strict inpainting at wrapper level.
            # Known/retained pixels remain exactly unchanged.
            mask_4d = removal_mask.unsqueeze(0).unsqueeze(0)
            attacked = (
                encoded_single * (1.0 - mask_4d)
                + reconstructed * mask_4d
            )

            attacked_images.append(attacked)
            used_removal_masks.append(removal_mask.detach().cpu())

        self.next_mask_index = end
        self.last_masks = used_removal_masks

        return [torch.cat(attacked_images, dim=0), cover_images]


def load_model_for_external_eval(
    hidden_config,
    checkpoint,
    device,
    external_noiser,
):
    """Load trained encoder/decoder while replacing the training noiser."""

    model = Hidden(
        hidden_config,
        device,
        external_noiser,
        tb_logger=None,
    )

    checkpoint_state = checkpoint["enc-dec-model"]

    filtered_state = {
        key: value
        for key, value in checkpoint_state.items()
        if not key.startswith("noiser.")
    }

    load_result = model.encoder_decoder.load_state_dict(
        filtered_state,
        strict=False,
    )

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

    if "discrim-model" in checkpoint:
        model.discriminator.load_state_dict(checkpoint["discrim-model"])

    model.encoder_decoder.eval()
    if hasattr(model, "discriminator"):
        model.discriminator.eval()

    return model


def to_float(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return float(value)


def save_mask_visual(mask_bool, output_path):
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    mask_tensor = torch.from_numpy(mask_uint8).float().unsqueeze(0) / 255.0
    vutils.save_image(mask_tensor, output_path)


def main():
    # Keep CPU for direct comparability with the current external sweep.
    device = torch.device("cpu")

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HiDDeN using precomputed optimized sparse masks "
            "with Mean, Telea, Navier-Stokes, or homogeneous diffusion."
        )
    )

    parser.add_argument("-d", "--data-dir", required=True)
    parser.add_argument(
        "-r", "--runs-root", "--runs_root",
        dest="runs_root", default="./runs"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mask-file", default=DEFAULT_MASK_FILE)
    parser.add_argument(
        "--attack",
        required=True,
        choices=list(FILL_MAP.keys()),
    )
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--diffusion-iterations", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--message-seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        default="./optimized_mask_eval_outputs_metrics",
    )
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--save-image-limit",
        type=int,
        default=10,
        help="Number of samples to save; -1 saves all.",
    )
    parser.add_argument("--save-mask-visuals", action="store_true")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load run config and checkpoint
    # ---------------------------------------------------------
    run_path = os.path.join(args.runs_root, args.run_name)
    options_file = os.path.join(run_path, "options-and-config.pickle")

    train_options, hidden_config, _ = utils.load_options(options_file)
    train_options.validation_folder = os.path.join(args.data_dir, "val")
    train_options.batch_size = args.batch_size

    checkpoint, checkpoint_name = utils.load_last_checkpoint(
        os.path.join(run_path, "checkpoints")
    )

    # ---------------------------------------------------------
    # Load optimized masks
    # ---------------------------------------------------------
    masks = np.load(args.mask_file, mmap_mode="r")
    if masks.ndim != 3:
        raise ValueError(
            "Expected masks with shape [N, H, W], "
            f"got {masks.shape}"
        )

    mean_retained_density = float(masks.mean())
    mean_removal_ratio = float(1.0 - mean_retained_density)

    # ---------------------------------------------------------
    # Output paths
    # ---------------------------------------------------------
    model_name = args.run_name.split(" ")[0]
    mask_stem = os.path.splitext(
        os.path.basename(args.mask_file)
    )[0]

    mask_set_name = os.path.basename(
        os.path.dirname(args.mask_file)
    )

    output_dir = os.path.join(
        args.output_root,
        (
            f"M_{model_name}"
            f"__A_{args.attack}"
            f"__{mask_set_name}"
            f"__{mask_stem}"
            f"__seed_{args.message_seed}"
        ),
    )
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(output_dir, "images")
    attacked_dir = os.path.join(images_dir, "attacked")
    cover_dir = os.path.join(images_dir, "cover")
    encoded_dir = os.path.join(images_dir, "encoded")
    masked_dir = os.path.join(images_dir, "masked")
    mask_vis_dir = os.path.join(images_dir, "masks")

    if args.save_images:
        os.makedirs(attacked_dir, exist_ok=True)
        os.makedirs(cover_dir, exist_ok=True)
        os.makedirs(encoded_dir, exist_ok=True)
        os.makedirs(masked_dir, exist_ok=True)
        if args.save_mask_visuals:
            os.makedirs(mask_vis_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "evaluation_results.json")
    per_image_path = os.path.join(output_dir, "per_image_metrics.json")
    run_config_path = os.path.join(output_dir, "run_config.json")

    # ---------------------------------------------------------
    # Build external attack and load model safely
    # ---------------------------------------------------------
    fill_strategy = build_fill_strategy(
        name=args.attack,
        tau=args.tau,
        diffusion_iterations=args.diffusion_iterations,
    )

    external_attack = PrecomputedOptimizedMaskInpainting(
        mask_file=args.mask_file,
        fill_strategy=fill_strategy,
    )

    model = load_model_for_external_eval(
        hidden_config=hidden_config,
        checkpoint=checkpoint,
        device=device,
        external_noiser=external_attack,
    )

    # ---------------------------------------------------------
    # Validation data + fixed messages
    # ---------------------------------------------------------
    _, val_data = utils.get_data_loaders(hidden_config, train_options)
    dataset_size = len(val_data.dataset)

    if len(masks) != dataset_size:
        raise ValueError(
            "Number of precomputed masks does not match validation dataset: "
            f"masks={len(masks)}, dataset={dataset_size}"
        )

    message_generator = torch.Generator(device="cpu")
    message_generator.manual_seed(args.message_seed)

    fixed_messages = torch.randint(
        low=0,
        high=2,
        size=(dataset_size, hidden_config.message_length),
        generator=message_generator,
        dtype=torch.float32,
    )

    print("=" * 60)
    print("OPTIMIZED SPARSE MASK EVALUATION")
    print("=" * 60)
    print(f"Run: {args.run_name}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Attack: {args.attack}")
    print(f"Mask file: {args.mask_file}")
    print(f"Mask shape: {masks.shape}")
    print(f"Mean retained density: {mean_retained_density:.6f}")
    print(f"Mean removal ratio: {mean_removal_ratio:.6f}")
    if args.attack == "diffusion":
        print(f"Diffusion iterations: {args.diffusion_iterations}")
        print(f"Tau: {args.tau}")
    print(f"Message seed: {args.message_seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    attack_psnr_meter = AverageMeter()
    attack_ssim_meter = AverageMeter()
    ber_meter = AverageMeter()
    actual_removal_meter = AverageMeter()

    external_attack.reset()
    processed_images = 0
    message_offset = 0
    per_image_records = []
    saved_visualizations = 0

    with torch.no_grad():
        for image, _ in val_data:
            image = image.to(device=device, dtype=torch.float32)
            current_batch_size = image.size(0)

            batch_mask_start = external_attack.next_mask_index
            batch_retention_masks = external_attack.masks[
                batch_mask_start : batch_mask_start + current_batch_size
            ]

            message = fixed_messages[
                message_offset : message_offset + current_batch_size
            ].to(device)
            message_offset += current_batch_size

            losses, (
                encoded_images,
                noised_images,
                decoded_messages,
            ) = model.validate_on_batch([image, message])

            cover = torch.clamp(convert_img_range(image), 0.0, 1.0)
            encoded = torch.clamp(
                convert_img_range(encoded_images), 0.0, 1.0
            )
            attacked = torch.clamp(
                convert_img_range(noised_images), 0.0, 1.0
            )

            # Final image quality: original cover vs attacked image.
            batch_psnr = compute_psnr(cover, attacked).item()
            batch_ssim = compute_ssim(cover, attacked).item()

            # Attack-only distortion: encoded image vs attacked image.
            batch_attack_psnr = compute_psnr(encoded, attacked).item()
            batch_attack_ssim = compute_ssim(encoded, attacked).item()

            ber_value = to_float(losses["bitwise-error  "])

            psnr_meter.update(batch_psnr, current_batch_size)
            ssim_meter.update(batch_ssim, current_batch_size)
            attack_psnr_meter.update(batch_attack_psnr, current_batch_size)
            attack_ssim_meter.update(batch_attack_ssim, current_batch_size)
            ber_meter.update(ber_value, current_batch_size)

            if external_attack.last_masks is not None:
                for removal_mask in external_attack.last_masks:
                    actual_removal_meter.update(
                        float(removal_mask.float().mean().item())
                    )

            for local_idx in range(current_batch_size):
                global_idx = processed_images + local_idx

                single_cover = cover[local_idx : local_idx + 1]
                single_encoded = encoded[local_idx : local_idx + 1]
                single_attacked = attacked[local_idx : local_idx + 1]

                # Final image quality: cover vs attacked.
                single_psnr = compute_psnr(
                    single_cover, single_attacked
                ).item()
                single_ssim = compute_ssim(
                    single_cover, single_attacked
                ).item()

                # Attack-only distortion: encoded vs attacked.
                single_attack_psnr = compute_psnr(
                    single_encoded, single_attacked
                ).item()
                single_attack_ssim = compute_ssim(
                    single_encoded, single_attacked
                ).item()

                retained_density = float(
                    batch_retention_masks[local_idx].mean()
                )

                per_image_records.append(
                    {
                        "index": int(global_idx),
                        "retained_density": retained_density,
                        "removal_ratio": 1.0 - retained_density,

                        # Legacy aliases kept for existing plotting code.
                        "psnr": float(single_psnr),
                        "ssim": float(single_ssim),

                        # Explicit metric names.
                        "psnr_final": float(single_psnr),
                        "ssim_final": float(single_ssim),
                        "psnr_attack": float(single_attack_psnr),
                        "ssim_attack": float(single_attack_ssim),

                        # Exact per-image BER when batch_size=1.
                        "ber": float(ber_value),
                        "ber_batch_value": float(ber_value),
                    }
                )

                if args.save_images:
                    save_all = args.save_image_limit == -1
                    if save_all or saved_visualizations < args.save_image_limit:
                        prefix = f"{global_idx:04d}"

                        vutils.save_image(
                            single_cover[0],
                            os.path.join(cover_dir, f"{prefix}_cover.png"),
                        )
                        vutils.save_image(
                            encoded[local_idx],
                            os.path.join(encoded_dir, f"{prefix}_encoded.png"),
                        )
                        vutils.save_image(
                            single_attacked[0],
                            os.path.join(attacked_dir, f"{prefix}_attacked.png"),
                        )

                        retention_tensor = torch.from_numpy(
                            batch_retention_masks[local_idx].astype(np.float32)
                        ).unsqueeze(0)

                        masked = encoded[local_idx].cpu() * retention_tensor
                        vutils.save_image(
                            masked,
                            os.path.join(masked_dir, f"{prefix}_masked.png"),
                        )

                        if args.save_mask_visuals:
                            retention_np = batch_retention_masks[local_idx]
                            removal_np = ~retention_np

                            save_mask_visual(
                                retention_np,
                                os.path.join(
                                    mask_vis_dir,
                                    f"{prefix}_retention_mask.png",
                                ),
                            )
                            save_mask_visual(
                                removal_np,
                                os.path.join(
                                    mask_vis_dir,
                                    f"{prefix}_removal_mask.png",
                                ),
                            )

                        saved_visualizations += 1

            processed_images += current_batch_size

            if processed_images % 50 == 0 or processed_images == dataset_size:
                print(f"Processed {processed_images}/{dataset_size}")

    # ---------------------------------------------------------
    # Sanity checks + result files
    # ---------------------------------------------------------
    if external_attack.next_mask_index != dataset_size:
        raise RuntimeError(
            "Mask consumption count does not match dataset size: "
            f"used={external_attack.next_mask_index}, dataset={dataset_size}"
        )

    if message_offset != dataset_size:
        raise RuntimeError(
            "Message consumption count does not match dataset size: "
            f"used={message_offset}, dataset={dataset_size}"
        )

    measured_removal_ratio = (
        float(actual_removal_meter.avg)
        if actual_removal_meter.count > 0
        else None
    )

    result = {
        "run_name": args.run_name,
        "model_name": model_name,
        "checkpoint": str(checkpoint_name),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "attack": args.attack,
        "mask_type": "optimized_sparse_ps_nlpe",
        "mask_file": args.mask_file,
        "mask_count": int(len(masks)),
        "mask_shape": list(masks.shape),
        "stored_mask_convention": (
            "1 = retained/known, 0 = removed/reconstructed"
        ),
        "fill_mask_convention": (
            "1 = removed/reconstructed, 0 = retained/known"
        ),
        "mean_retained_density": mean_retained_density,
        "mean_removal_ratio": mean_removal_ratio,
        "measured_removal_ratio": measured_removal_ratio,
        "tau": args.tau if args.attack == "diffusion" else None,
        "diffusion_iterations": (
            args.diffusion_iterations if args.attack == "diffusion" else None
        ),
        "message_seed": args.message_seed,
        "batch_size": args.batch_size,
        "processed_images": processed_images,

        # Legacy aliases kept for existing plotting code.
        "psnr": float(psnr_meter.avg),
        "ssim": float(ssim_meter.avg),

        # Explicit image-quality metrics.
        "psnr_final": float(psnr_meter.avg),
        "ssim_final": float(ssim_meter.avg),
        "psnr_attack": float(attack_psnr_meter.avg),
        "ssim_attack": float(attack_ssim_meter.avg),

        "ber": float(ber_meter.avg),
        "saved_visualizations": int(saved_visualizations),
        "output_dir": output_dir,
        "files": {
            "summary_json": summary_path,
            "per_image_metrics_json": per_image_path,
            "run_config_json": run_config_path,
        },
    }

    run_config = {
        "data_dir": args.data_dir,
        "runs_root": args.runs_root,
        "run_name": args.run_name,
        "mask_file": args.mask_file,
        "attack": args.attack,
        "tau": args.tau,
        "diffusion_iterations": args.diffusion_iterations,
        "batch_size": args.batch_size,
        "message_seed": args.message_seed,
        "output_root": args.output_root,
        "save_images": args.save_images,
        "save_image_limit": args.save_image_limit,
        "save_mask_visuals": args.save_mask_visuals,
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    with open(per_image_path, "w", encoding="utf-8") as file:
        json.dump(per_image_records, file, indent=4, ensure_ascii=False)

    with open(run_config_path, "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=4, ensure_ascii=False)

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Retained density = {mean_retained_density:.6f}")
    print(f"Removal ratio    = {mean_removal_ratio:.6f}")
    print(f"PSNR final  (cover -> attacked)  = {psnr_meter.avg:.4f}")
    print(f"SSIM final  (cover -> attacked)  = {ssim_meter.avg:.4f}")
    print(f"PSNR attack (encoded -> attacked) = {attack_psnr_meter.avg:.4f}")
    print(f"SSIM attack (encoded -> attacked) = {attack_ssim_meter.avg:.4f}")
    print(f"BER                              = {ber_meter.avg:.6f}")
    print("-" * 60)
    print(f"Saved output directory -> {output_dir}")
    print(f"Summary JSON          -> {summary_path}")
    print(f"Per-image metrics     -> {per_image_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
