import os
import torch
import torch.nn as nn

from noise_layers.pretrained_unet_inpainting import (
    SmallUNet,
    ControlledRectangleMaskGenerator,
)


def _extract_state_dict(checkpoint):
    """
    Support several common checkpoint formats.
    """

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    # Common wrapped checkpoint formats
    for key in (
        "model_state_dict",
        "unet_state_dict",
        "state_dict",
        "model",
        "unet",
    ):
        if key not in checkpoint:
            continue

        value = checkpoint[key]

        if isinstance(value, nn.Module):
            return value.state_dict()

        if isinstance(value, dict):
            return value

    # Maybe the checkpoint itself is already a raw state_dict
    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise KeyError(
        "Could not find a U-Net state_dict in checkpoint. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def _strip_prefix_if_needed(state_dict):
    """
    Remove common wrapper prefixes if present.
    """

    prefixes = (
        "module.",
        "unet.",
        "model.",
    )

    result = state_dict

    for prefix in prefixes:
        if result and all(
            key.startswith(prefix)
            for key in result.keys()
        ):
            result = {
                key[len(prefix):]: value
                for key, value in result.items()
            }

    return result


class FrozenPretrainedUNetInpainting(nn.Module):
    """
    Frozen pretrained U-Net used as a differentiable
    inpainting-style corruption layer during HiDDeN training.

    Important:
    - U-Net weights are pretrained for image reconstruction.
    - U-Net parameters are frozen during HiDDeN training.
    - Gradients are still allowed to propagate THROUGH the U-Net
      to the encoded image and therefore to the watermark encoder.
    - Mask convention:
          0 = known / retained
          1 = missing / reconstructed
    """

    def __init__(
        self,
        checkpoint_path,
        min_ratio=0.1,
        max_ratio=0.4,
        base_channels=32,
        min_mask_size=8,
        max_aspect_ratio=4.0,
        seed=42,
        randomize_ratio=True,
    ):
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.base_channels = base_channels

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"U-Net checkpoint not found: {checkpoint_path}"
            )

        # --------------------------------------------------
        # Reconstruction model
        # --------------------------------------------------
        self.unet = SmallUNet(
            base_channels=base_channels
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = _extract_state_dict(checkpoint)
        state_dict = _strip_prefix_if_needed(state_dict)

        self.unet.load_state_dict(
            state_dict,
            strict=True,
        )

        # --------------------------------------------------
        # Freeze U-Net
        # --------------------------------------------------
        for parameter in self.unet.parameters():
            parameter.requires_grad_(False)

        self.unet.eval()

        # --------------------------------------------------
        # Controlled exact-coverage mask generator
        # --------------------------------------------------
        self.mask_generator = ControlledRectangleMaskGenerator(
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            min_mask_size=min_mask_size,
            max_aspect_ratio=max_aspect_ratio,
            seed=seed,
            randomize_ratio=randomize_ratio,
        )

        # Only for debugging / sanity checks
        self.last_mask = None

    def train(self, mode=True):
        """
        HiDDeN will call .train() on EncoderDecoder.
        Keep the frozen U-Net itself in eval mode.
        """
        super().train(mode)
        self.unet.eval()
        return self

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        B, C, H, W = noised_image.shape

        mask = self.mask_generator.generate(
            B=B,
            H=H,
            W=W,
            device=noised_image.device,
        )

        self.last_mask = mask.detach()

        # --------------------------------------------------
        # CRITICAL:
        # NO torch.no_grad()
        # NO detach()
        #
        # U-Net parameters are frozen, but autograd must
        # differentiate the output with respect to the input.
        # --------------------------------------------------
        corrupted_image = self.unet(
            noised_image,
            mask,
        )

        return [
            corrupted_image,
            cover_image,
        ]