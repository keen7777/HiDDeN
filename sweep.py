import os
import argparse
import torch
import numpy as np
import json

import utils
from model.hidden import Hidden
from average_meter import AverageMeter
from evaluation.rgb_yuv_convert import convert_img_range
from evaluation.psnr_ssim_evaluation import compute_psnr, compute_ssim

from noise_layers.dropout import Dropout
from noise_layers.crop import Crop
from noise_layers.resize import Resize
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.mask_inpainting import MaskInpainting
from noise_layers.mask_inpainting_telea import MaskInpaintingTelea


# =========================
# BUILD NOISE LAYER (FIXED)
# =========================
def build_noise_layer(name, s):

    
    # s means attacking strength

    if name == "dropout":
        # FIXED version: deterministic sweep
        return Dropout((s, s))

    if name == "crop":
        return Crop((s, s), (s, s))

    if name == "resize":
        return Resize((s, s))
    
    if name == "maskinpainting":
        return MaskInpainting(s,s)
    
    if name == "teleamaskinpainting":
        return MaskInpaintingTelea(s,s)

    if name == "jpeg":
        # IMPORTANT: map strength -> keep coefficients
        # higher s = stronger compression
        s = 1 - s
        keep = int(64 * s)
        keep = max(1, min(64, keep))

        return JpegCompression(
            device="cpu",
            yuv_keep_weights=(keep, max(1, keep // 3), max(1, keep // 3))
        )

    raise ValueError(f"Unknown attack: {name}")


# =========================
# MAIN
# =========================
def main():

    device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", required=True)
    parser.add_argument("-r", "--runs_root", default="./runs")
    parser.add_argument("--run-name", required=True)

    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument("--min", type=float, default=0.1)
    parser.add_argument("--max", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=10)

    args = parser.parse_args()

    run_path = os.path.join(args.runs_root, args.run_name)

    options_file = os.path.join(run_path, "options-and-config.pickle")
    train_options, hidden_config, noise_config = utils.load_options(options_file)

    train_options.validation_folder = os.path.join(args.data_dir, "val")

    checkpoint, _ = utils.load_last_checkpoint(os.path.join(run_path, "checkpoints"))

    strengths = np.linspace(args.min, args.max, args.steps)

    results = {}

    for s in strengths:

        print(f"\n===== {args.attack} | strength={s:.3f} =====")

        noise_layer = build_noise_layer(args.attack, s)
        noiser = noise_layer

        model = Hidden(hidden_config, device, noiser, tb_logger=None)
        utils.model_from_checkpoint(model, checkpoint)

        _, val_data = utils.get_data_loaders(hidden_config, train_options)

        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        ber_meter = AverageMeter()

        for image, _ in val_data:

            image = image.to(device)

            message = torch.randint(
                0, 2,
                (image.size(0), hidden_config.message_length)
            ).float().to(device)

            losses, (encoded_images, noised_images, decoded) = model.validate_on_batch([image, message])

            cover = convert_img_range(image)
            noised_images = convert_img_range(noised_images)

            psnr_meter.update(compute_psnr(cover, noised_images).item())
            ssim_meter.update(compute_ssim(cover, noised_images).item())

            ber_meter.update(losses["bitwise-error  "])

        results[float(s)] = {
            "psnr": psnr_meter.avg,
            "ssim": ssim_meter.avg,
            "ber": ber_meter.avg
        }

        print(f"PSNR={psnr_meter.avg:.4f}, SSIM={ssim_meter.avg:.4f}, BER={ber_meter.avg:.6f}")

    out_file = f"sweep_{args.run_name}_{args.attack}.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()