import cv2
import numpy as np
import torch

from .base import FillStrategy


class ShiftMapFill(FillStrategy):
    """
    Patch-correspondence-based inpainting using
    OpenCV xphoto ShiftMap.

    Mask convention expected by this project:
        removal_mask = 1 -> missing / reconstruct
        removal_mask = 0 -> known / retain

    Important:
    OpenCV xphoto uses the opposite convention:
        non-zero -> valid
        zero     -> inpaint
    """

    def fill(self, image, removal_mask):
        if not hasattr(cv2, "xphoto"):
            raise RuntimeError(
                "OpenCV xphoto is unavailable. "
                "Install/use an OpenCV build with contrib modules."
            )

        # image: [1, 3, H, W], normally [-1, 1]
        img = image[0].detach()

        if img.min() < 0:
            img = (img + 1.0) / 2.0

        img = torch.clamp(
            img,
            0.0,
            1.0,
        )

        rgb = (
            img.permute(1, 2, 0)
            .cpu()
            .numpy()
            * 255.0
        ).astype(np.uint8)

        removal_np = (
            removal_mask.detach()
            .cpu()
            .numpy()
            > 0.5
        )

        # ----------------------------------------
        # xphoto has the OPPOSITE mask convention:
        #
        # 255 = valid pixel
        #   0 = pixel to reconstruct
        # ----------------------------------------
        valid_mask = (
            (~removal_np).astype(np.uint8)
            * 255
        )

        # ShiftMap documentation recommends Lab
        # or a similar intensity/chrominance space.
        lab = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2LAB,
        )

        # IMPORTANT:
        # actually erase missing pixels before giving
        # the image to the reconstruction algorithm.
        #
        # Therefore there is no ground-truth leakage.
        masked_lab = lab.copy()
        # Neutral black in OpenCV uint8 Lab:
        # L = 0
        # a = 128
        # b = 128
        masked_lab[removal_np] = np.array(
            [0, 128, 128],
            dtype=np.uint8,
        )

        reconstructed_lab = np.empty_like(
            masked_lab
        )

        cv2.xphoto.inpaint(
            masked_lab,
            valid_mask,
            reconstructed_lab,
            cv2.xphoto.INPAINT_SHIFTMAP,
        )

        reconstructed_rgb = cv2.cvtColor(
            reconstructed_lab,
            cv2.COLOR_LAB2RGB,
        )

        reconstructed = (
            torch.from_numpy(
                reconstructed_rgb
            )
            .float()
            / 255.0
        )

        reconstructed = reconstructed.permute(
            2, 0, 1
        )

        reconstructed = (
            reconstructed * 2.0 - 1.0
        ).to(image.device)

        # Strict inpainting:
        # known pixels remain exactly unchanged.
        mask = removal_mask.to(
            device=image.device,
            dtype=image.dtype,
        ).unsqueeze(0)

        output = (
            image[0] * (1.0 - mask)
            + reconstructed * mask
        )

        return output.unsqueeze(0)