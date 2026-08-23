import numpy as np
import torch.nn as nn

from noise_layers.mask_generator import RectangleRemovalMaskGenerator


class EvalInpainting(nn.Module):
    """
    Evaluation-only inpainting attack.

    Mask convention:
        removal_mask = 1 -> removed / reconstructed
        removal_mask = 0 -> retained / known

    The mask generator and fill strategy are separated so that
    different inpainting methods can be evaluated with the same
    spatial mask distribution.
    """

    def __init__(
        self,
        max_mask_ratio=1.0,
        min_mask_ratio=0.05,
        max_mask_number=50,
        min_mask_size=8,
        max_aspect_ratio=3.0,
        fill_strategy=None,
        seed=42,
        randomize_ratio=True,
    ):
        super(EvalInpainting, self).__init__()

        self.max_mask_ratio = max_mask_ratio
        self.min_mask_ratio = min_mask_ratio
        self.randomize_ratio = randomize_ratio

        if fill_strategy is None:
            raise ValueError(
                "EvalInpainting requires a fill_strategy."
            )

        self.fill_strategy = fill_strategy

        # Separate RNG for sampling the removal ratio.
        self.ratio_rng = np.random.RandomState(seed)

        # Independent RNG for mask geometry.
        # This prevents randomness inside the fill strategy
        # from affecting the generated masks.
        mask_seed = (
            None if seed is None
            else seed + 1
        )

        self.mask_generator = (
            RectangleRemovalMaskGenerator(
                min_mask_size=min_mask_size,
                max_aspect_ratio=max_aspect_ratio,
                max_rectangles=max_mask_number,
                max_rectangle_ratio=0.10,
                margin=1,
                seed=mask_seed,
            )
        )

        # Useful for debugging / visualisation.
        self.last_masks = None
        self.last_removal_ratios = None

    def _sample_removal_ratio(self):
        if self.randomize_ratio:
            return float(
                self.ratio_rng.uniform(
                    self.min_mask_ratio,
                    self.max_mask_ratio,
                )
            )

        return float(
            self.max_mask_ratio
        )

    def forward(self, noised_and_cover):
        noised_image, cover_image = noised_and_cover

        B, C, H, W = noised_image.shape

        output = noised_image.clone()

        masks = []
        removal_ratios = []

        for b in range(B):
            removal_ratio = (
                self._sample_removal_ratio()
            )

            removal_mask = (
                self.mask_generator.generate(
                    batch_size=1,
                    H=H,
                    W=W,
                    removal_ratio=removal_ratio,
                    device=output.device,
                )[0, 0]
            )

            filled = self.fill_strategy.fill(
                output[b:b + 1],
                removal_mask,
            )

            output[b:b + 1] = filled

            masks.append(
                removal_mask.detach()
            )

            removal_ratios.append(
                removal_ratio
            )

        self.last_masks = masks
        self.last_removal_ratios = (
            removal_ratios
        )

        return [
            output,
            cover_image,
        ]