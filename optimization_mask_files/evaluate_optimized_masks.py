import os
import sys
import argparse
import json
from pathlib import Path

# Add the HiDDeN project root to Python's import path.
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
from mask_optimization import inpaint_hom_diff


DEFAULT_MASK_FILE = (
    "/home/keen/HiDDeN/evaluation_data/"
    "opt_masks_diff_150_nlpe_100_den_0.1/"
    "val_ps_nlpe_masks_density_100.npy"
)


class PrecomputedDiffusionInpainting(nn.Module):
    """
    HiDDeN-compatible noise layer using precomputed optimized masks.

    Mask convention:
        True  = known / retained pixel
        False = missing / inpainted pixel

    The masks are consumed in validation-dataset order.
    """

    def __init__(
        self,
        mask_file,
        tau=0.25,
        diffusion_iterations=150,
    ):
        super().__init__()

        self.mask_file = mask_file
        self.masks = np.load(mask_file)

        if self.masks.ndim != 3:
            raise ValueError(
                "Expected mask array with shape [N, H, W], "
                f"got {self.masks.shape}"
            )

        self.masks = self.masks.astype(np.bool_, copy=False)
        self.tau = tau
        self.diffusion_iterations = diffusion_iterations
        self.next_mask_index = 0

    def reset(self):
        self.next_mask_index = 0

    def current_batch_indices(self, batch_size):
        start = self.next_mask_index
        end = start + batch_size
        if end > len(self.masks):
            raise IndexError(
                "Not enough precomputed masks. "
                f"Requested masks [{start}:{end}], "
                f"but only {len(self.masks)} masks are available."
            )
        return start, end

    def get_mask_batch(self, batch_size):
        start, end = self.current_batch_indices(batch_size)
        return self.masks[start:end], start, end

    def forward(self, encoded_and_cover):
        if not isinstance(encoded_and_cover, (list, tuple)):
            raise TypeError(
                "Expected [encoded_images, cover_images] as input."
            )

        if len(encoded_and_cover) != 2:
            raise ValueError(
                "Expected exactly two inputs: "
                "[encoded_images, cover_images]."
            )

        encoded_images, cover_images = encoded_and_cover
        batch_size = encoded_images.size(0)
        batch_masks, start, end = self.get_mask_batch(batch_size)

        noised_images = []

        # HiDDeN tensors are normally in [-1, 1].
        # Convert each encoded image to [0, 1] before NumPy diffusion,
        # then convert the reconstruction back to [-1, 1].
        for batch_index in range(batch_size):
            encoded = encoded_images[batch_index]

            image_01 = ((encoded.detach().cpu() + 1.0) / 2.0)
            image_01 = torch.clamp(image_01, 0.0, 1.0)

            image_np = (
                image_01
                .permute(1, 2, 0)
                .numpy()
                .astype(np.float64, copy=False)
            )

            mask = batch_masks[batch_index]

            if mask.shape != image_np.shape[:2]:
                raise ValueError(
                    "Mask/image shape mismatch: "
                    f"mask={mask.shape}, image={image_np.shape[:2]}"
                )

            reconstructed = inpaint_hom_diff(
                known_image_data=image_np,
                mask=mask,
                num_iterations=self.diffusion_iterations,
                tau=self.tau,
            )

            reconstructed = np.clip(
                reconstructed,
                0.0,
                1.0,
            ).astype(np.float32)

            reconstructed_tensor = torch.from_numpy(
                reconstructed
            ).permute(2, 0, 1)

            reconstructed_tensor = (
                reconstructed_tensor * 2.0 - 1.0
            ).to(
                device=encoded_images.device,
                dtype=encoded_images.dtype,
            )

            noised_images.append(reconstructed_tensor)

        self.next_mask_index = end
        noised_images = torch.stack(noised_images, dim=0)
        return [noised_images, cover_images]


def save_mask_visual(mask_bool, output_path):
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    mask_tensor = torch.from_numpy(mask_uint8).float().unsqueeze(0) / 255.0
    vutils.save_image(mask_tensor, output_path)



def main():
    device = torch.device("cpu")

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HiDDeN using precomputed optimized PS+NLPE masks "
            "and save results into a dedicated output folder."
        )
    )

    parser.add_argument("-d", "--data-dir", required=True)
    parser.add_argument("-r", "--runs_root", default="./runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mask-file", default=DEFAULT_MASK_FILE)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--diffusion-iterations", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--message-seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        default="./optimized_mask_eval_outputs",
        help="Root directory where a dedicated output folder will be created.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save attacked-image visualizations.",
    )
    parser.add_argument(
        "--save-image-limit",
        type=int,
        default=10,
        help="How many validation samples to visualize. Use -1 for all.",
    )
    parser.add_argument(
        "--save-mask-visuals",
        action="store_true",
        help="Also save the corresponding mask PNG visualizations.",
    )

    args = parser.parse_args()

    run_path = os.path.join(args.runs_root, args.run_name)
    options_file = os.path.join(run_path, "options-and-config.pickle")
    train_options, hidden_config, _ = utils.load_options(options_file)

    train_options.validation_folder = os.path.join(args.data_dir, "val")
    train_options.batch_size = args.batch_size

    checkpoint, checkpoint_name = utils.load_last_checkpoint(
        os.path.join(run_path, "checkpoints")
    )

    masks = np.load(args.mask_file, mmap_mode="r")
    if masks.ndim != 3:
        raise ValueError(
            "Expected masks with shape [N, H, W], "
            f"got {masks.shape}"
        )

    model_name = args.run_name.split(" ")[0]
    mask_stem = os.path.splitext(os.path.basename(args.mask_file))[0]
    output_dir = os.path.join(
        args.output_root,
        f"M_{model_name}__{mask_stem}__diff_{args.diffusion_iterations}__seed_{args.message_seed}",
    )
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(output_dir, "images")
    attacked_dir = os.path.join(images_dir, "attacked")
    cover_dir = os.path.join(images_dir, "cover")
    encoded_dir = os.path.join(images_dir, "encoded")
    mask_vis_dir = os.path.join(images_dir, "masks")

    if args.save_images:
        os.makedirs(attacked_dir, exist_ok=True)
        os.makedirs(cover_dir, exist_ok=True)
        os.makedirs(encoded_dir, exist_ok=True)
        if args.save_mask_visuals:
            os.makedirs(mask_vis_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "evaluation_results.json")
    per_image_path = os.path.join(output_dir, "per_image_metrics.json")
    run_config_path = os.path.join(output_dir, "run_config.json")

    print("====================================")
    print("OPTIMIZED MASK EVALUATION")
    print("------------------------------------")
    print(f"Run: {args.run_name}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Mask file: {args.mask_file}")
    print(f"Mask shape: {masks.shape}")
    print(f"Mean retained density: {float(masks.mean()):.6f}")
    print(f"Diffusion iterations: {args.diffusion_iterations}")
    print(f"Tau: {args.tau}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {output_dir}")
    print("====================================")

    noiser = PrecomputedDiffusionInpainting(
        mask_file=args.mask_file,
        tau=args.tau,
        diffusion_iterations=args.diffusion_iterations,
    )

    model = Hidden(hidden_config, device, noiser, tb_logger=None)
    utils.model_from_checkpoint(model, checkpoint)

    _, val_data = utils.get_data_loaders(hidden_config, train_options)
    dataset_size = len(val_data.dataset)

    if len(masks) != dataset_size:
        raise ValueError(
            "Number of precomputed masks does not match "
            "validation dataset size: "
            f"masks={len(masks)}, dataset={dataset_size}"
        )

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    ber_meter = AverageMeter()

    message_generator = torch.Generator(device="cpu")
    message_generator.manual_seed(args.message_seed)

    noiser.reset()
    processed_images = 0
    per_image_records = []
    saved_visualizations = 0

    for batch_index, (image, _) in enumerate(val_data):
        image = image.to(device)
        batch_size = image.size(0)

        batch_mask_start = noiser.next_mask_index
        batch_masks = noiser.masks[
            batch_mask_start : batch_mask_start + batch_size
        ]

        message = torch.randint(
            0,
            2,
            (image.size(0), hidden_config.message_length),
            generator=message_generator,
        ).float().to(device)

        losses, (encoded_images, noised_images, decoded) = model.validate_on_batch(
            [image, message]
        )

        cover = torch.clamp(convert_img_range(image), 0, 1)
        encoded_01 = torch.clamp(convert_img_range(encoded_images), 0, 1)
        noised_01 = torch.clamp(convert_img_range(noised_images), 0, 1)

        batch_psnr = compute_psnr(cover, noised_01).item()
        batch_ssim = compute_ssim(cover, noised_01).item()

        psnr_meter.update(batch_psnr, batch_size)
        ssim_meter.update(batch_ssim, batch_size)
        ber_meter.update(losses["bitwise-error  "], batch_size)

        for local_idx in range(batch_size):
            global_idx = processed_images + local_idx

            single_cover = cover[local_idx : local_idx + 1]
            single_noised = noised_01[local_idx : local_idx + 1]
            single_psnr = compute_psnr(single_cover, single_noised).item()
            single_ssim = compute_ssim(single_cover, single_noised).item()

            per_image_records.append(
                {
                    "index": int(global_idx),
                    "psnr": float(single_psnr),
                    "ssim": float(single_ssim),
                    "ber_batch_value": float(losses["bitwise-error  "]),
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
                        encoded_01[local_idx],
                        os.path.join(encoded_dir, f"{prefix}_encoded.png"),
                    )
                    vutils.save_image(
                        single_noised[0],
                        os.path.join(attacked_dir, f"{prefix}_attacked.png"),
                    )
                    if args.save_mask_visuals:
                        save_mask_visual(
                            batch_masks[local_idx],
                            os.path.join(mask_vis_dir, f"{prefix}_mask.png"),
                        )
                    saved_visualizations += 1

        processed_images += batch_size

        if processed_images % 50 == 0 or processed_images == dataset_size:
            print(f"Processed {processed_images}/{dataset_size}")

    if noiser.next_mask_index != dataset_size:
        raise RuntimeError(
            "Mask consumption count does not match dataset size: "
            f"used={noiser.next_mask_index}, dataset={dataset_size}"
        )

    result = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_name),
        "mask_file": args.mask_file,
        "mask_count": int(len(masks)),
        "mask_shape": list(masks.shape),
        "mean_retained_density": float(masks.mean()),
        "tau": args.tau,
        "diffusion_iterations": args.diffusion_iterations,
        "message_seed": args.message_seed,
        "batch_size": args.batch_size,
        "processed_images": processed_images,
        "psnr": psnr_meter.avg,
        "ssim": ssim_meter.avg,
        "ber": ber_meter.avg,
        "saved_visualizations": int(saved_visualizations),
        "output_dir": output_dir,
        "files": {
            "summary_json": summary_path,
            "per_image_metrics_json": per_image_path,
        },
    }

    run_config = {
        "data_dir": args.data_dir,
        "runs_root": args.runs_root,
        "run_name": args.run_name,
        "mask_file": args.mask_file,
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
        json.dump(result, file, indent=4)

    with open(per_image_path, "w", encoding="utf-8") as file:
        json.dump(per_image_records, file, indent=4)

    with open(run_config_path, "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=4)

    print("\n====================================")
    print("EVALUATION RESULTS")
    print("------------------------------------")
    print(f"PSNR = {psnr_meter.avg:.4f}")
    print(f"SSIM = {ssim_meter.avg:.4f}")
    print(f"BER  = {ber_meter.avg:.6f}")
    print("------------------------------------")
    print(f"Saved output directory -> {output_dir}")
    print(f"Summary JSON          -> {summary_path}")
    print(f"Per-image metrics     -> {per_image_path}")
    if args.save_images:
        print(f"Image visualizations  -> {images_dir}")
    print("====================================")


if __name__ == "__main__":
    main()
