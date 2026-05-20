import os
import argparse
import torch
import numpy as np
import json

import utils

from model.hidden import Hidden
from noise_layers.noiser import Noiser
from noise_layers.dropout import Dropout
from noise_layers.crop import Crop
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.resize import Resize

from average_meter import AverageMeter
from evaluation.rgb_yuv_convert import convert_img_range, rgb_to_yuv
from evaluation.psnr_ssim_evaluation import compute_psnr, compute_ssim


# =========================
# FIXED NOISE WRAPPER
# =========================
class FixedNoiser(torch.nn.Module):
    """
    deterministic noise layer wrapper for sweep
    """
    def __init__(self, noise_layer):
        super().__init__()
        self.noise_layer = noise_layer

    def forward(self, x):
        return self.noise_layer(x)


# =========================
# BUILD NOISE LAYER
# =========================
def build_noise_layer(name, value):
    """
    value = attack strength
    """

    if name == "dropout":
        return Dropout((value, value))

    if name == "crop":
        return Crop((value, value), (value, value))

    if name == "resize":
        return Resize(value, value)

    if name == "jpeg":
        # jpeg usually quality = 1 - strength
        return JpegCompression(device="cpu", quality=int(value))

    raise ValueError(f"Unknown attack: {name}")


# =========================
# MAIN SWEEP
# =========================
def main():
    device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", required=True)
    parser.add_argument("-r", "--runs_root", default="./experiments")
    parser.add_argument("--run-name", type=str, required=True)

    parser.add_argument("--attack", type=str, default="dropout")
    parser.add_argument("--min", type=float, default=0.1)
    parser.add_argument("--max", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=9)

    args = parser.parse_args()

    # =========================
    # LOAD MODEL
    # =========================
    run_path = os.path.join(args.runs_root, args.run_name)

    options_file = os.path.join(run_path, "options-and-config.pickle")
    train_options, hidden_config, noise_config = utils.load_options(options_file)

    train_options.validation_folder = os.path.join(args.data_dir, "val")

    checkpoint, _ = utils.load_last_checkpoint(os.path.join(run_path, "checkpoints"))

    results = {}

    # =========================
    # SWEEP LOOP
    # =========================
    strengths = np.linspace(args.min, args.max, args.steps)

    for s in strengths:

        print(f"\n===== Attack {args.attack} | strength={s:.3f} =====")

        noise_layer = build_noise_layer(args.attack, s)
        noiser = FixedNoiser(noise_layer)

        model = Hidden(hidden_config, device, noiser, tb_logger=None)
        utils.model_from_checkpoint(model, checkpoint)

        _, val_data = utils.get_data_loaders(hidden_config, train_options)

        psnr_meter = AverageMeter()
        ber_meter = AverageMeter()
        ssim_meter = AverageMeter()

        for image, _ in val_data:

            image = image.to(device)
            message = torch.randint(
                0, 2,
                (image.size(0), hidden_config.message_length)
            ).float().to(device)

            losses, (encoded, _, decoded) = model.validate_on_batch([image, message])

            cover = convert_img_range(image)
            encoded = convert_img_range(encoded)

            psnr_meter.update(compute_psnr(cover, encoded).item())
            ssim_meter.update(compute_ssim(cover, encoded).item())

            ber_meter.update(losses["bitwise-error  "])

        results[float(s)] = {
            "psnr": psnr_meter.avg,
            "ssim": ssim_meter.avg,
            "ber": ber_meter.avg
        }

        print(f"PSNR: {psnr_meter.avg:.4f}, SSIM: {ssim_meter.avg:.4f}, BER: {ber_meter.avg:.6f}")

    # =========================
    # SAVE RESULTS
    # =========================
    out_file = f"sweep_{args.run_name}_{args.attack}.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved sweep results to {out_file}")


if __name__ == "__main__":
    main()