import numpy as np
import torch

from .base import FillStrategy
from optimization_mask_files.mask_optimization import inpaint_hom_diff


class DiffusionFill(FillStrategy):

    def __init__(
        self,
        tau=0.25,
        num_iterations=150,
    ):
        self.tau = tau
        self.num_iterations = num_iterations

    def fill(
        self,
        image,
        removal_mask,
    ):
        """
        image:
            [1, 3, H, W], normally [-1, 1]

        removal_mask:
            1 = removed / reconstruct
            0 = retained / known
        """

        img = image[0].detach()

        image_01 = torch.clamp(
            (img + 1.0) / 2.0,
            0.0,
            1.0,
        )

        image_np = (
            image_01
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        removal_np = (
            removal_mask.detach()
            .cpu()
            .numpy()
            > 0.5
        )

        # inpaint_hom_diff convention:
        # True = known / retained
        retention_mask = ~removal_np

        reconstructed = inpaint_hom_diff(
            known_image_data=image_np,
            mask=retention_mask,
            num_iterations=self.num_iterations,
            tau=self.tau,
        )

        reconstructed = np.clip(
            reconstructed,
            0.0,
            1.0,
        )

        # Strictly preserve known pixels.
        reconstructed[retention_mask] = (
            image_np[retention_mask]
        )

        reconstructed = (
            torch.from_numpy(
                reconstructed.astype(np.float32)
            )
            .permute(2, 0, 1)
        )

        reconstructed = (
            reconstructed * 2.0 - 1.0
        ).to(
            device=image.device,
            dtype=image.dtype,
        )

        mask = removal_mask.to(
            device=image.device,
            dtype=image.dtype,
        ).unsqueeze(0)

        output = (
            image[0] * (1.0 - mask)
            + reconstructed * mask
        )

        return output.unsqueeze(0)