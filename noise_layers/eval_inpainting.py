import torch
import torch.nn as nn
import numpy as np

from noise_layers.fill_strategies.mean_fill import MeanFill
from noise_layers.fill_strategies.random_fill import RandomNeighborFill
from noise_layers.fill_strategies.blur_fill import BlurFill
from noise_layers.fill_strategies.telea_fill import TeleaFill
from noise_layers.fill_strategies.patchmatch_fill import PatchMatchFill
from noise_layers.fill_strategies.resnet_fill import ResNetFill
from noise_layers.fill_strategies.diffusion_fill import DiffusionFill


class EvalInpainting(nn.Module):
    """
    Evaluation-only random multi-mask inpainting attack.

    Main idea:
    1. Randomly choose a TOTAL mask ratio
    2. Generate multiple random rectangles
    3. Merge them into one union mask
    4. Apply a real or evaluation fill strategy to the masked region

    Note:
    - This layer is mainly for eval / sweep.
    - The generated mask is a union mask.
    - The fill strategy receives the whole union mask once.
    """

    def __init__(
        self,
        max_mask_ratio=1,
        max_mask_number=10,
        min_mask_size=8,
        max_aspect_ratio=3.0,
        fill_strategy=None,
        seed=None,
        randomize_ratio=True,
    ):
        super(EvalInpainting, self).__init__()

        # Maximum TOTAL masked area ratio.
        # Example: 0.5 means the union mask covers at most 50% of the image.
        self.max_mask_ratio = max_mask_ratio

        # Maximum number of candidate rectangles.
        # Note: the actual number may be smaller if target coverage is reached early.
        self.max_mask_number = max_mask_number

        # Minimum side length for sampled rectangles.
        self.min_mask_size = min_mask_size

        # Maximum rectangle aspect ratio.
        # Example: 3.0 means width / height can vary between 1/3 and 3.
        self.max_aspect_ratio = max_aspect_ratio

        # Fill strategy, e.g. TeleaFill, PatchMatchFill, BlurFill, etc.
        self.fill_strategy = fill_strategy

        # Random generator.
        # If seed is None, the masks are fully random.
        self.rng = np.random.RandomState(seed)

        # If True, randomly sample mask ratio in [0.05, max_mask_ratio].
        # If False, use max_mask_ratio directly. This is useful for sweep.
        self.randomize_ratio = randomize_ratio

    def generate_union_mask(self, H, W, device):
        """
        Generate one union mask with target pixel-level coverage.

        Returns:
            mask: torch.Tensor, shape [H, W], values in {0.0, 1.0}
        """

        if self.randomize_ratio:
            target_coverage = self.rng.uniform(0.05, self.max_mask_ratio)
        else:
            target_coverage = self.max_mask_ratio

        # Pixel-level union mask.
        mask_union = np.zeros((H, W), dtype=np.float32)

        # Keep a small margin to avoid boundary issues.
        max_w = max(1, W - 4)
        max_h = max(1, H - 4)

        max_iters = 300
        iters = 0
        num_masks = 0

        while (
            mask_union.mean() < target_coverage
            and iters < max_iters
            and num_masks < self.max_mask_number
        ):
            iters += 1

            aspect_ratio = self.rng.uniform(
                1.0 / self.max_aspect_ratio,
                self.max_aspect_ratio
            )

            # Candidate rectangle area.
            # The upper bound prevents one rectangle from becoming too dominant.
            area = self.rng.uniform(
                self.min_mask_size ** 2,
                H * W * 0.1
            )

            w = int(np.sqrt(area * aspect_ratio))
            h = int(np.sqrt(area / aspect_ratio))

            w = max(1, min(w, max_w))
            h = max(1, min(h, max_h))

            # Skip invalid rectangles.
            if max_w - w <= 1 or max_h - h <= 1:
                continue

            left = self.rng.randint(1, max_w - w)
            top = self.rng.randint(1, max_h - h)

            # Add this rectangle to the union mask.
            mask_union[top:top + h, left:left + w] = 1.0
            num_masks += 1

        mask = torch.from_numpy(mask_union).to(device=device)

        return mask

    def forward(self, noised_and_cover):
        noised_image, cover_image = noised_and_cover

        B, C, H, W = noised_image.shape
        output = noised_image.clone()

        for b in range(B):
            mask = self.generate_union_mask(
                H=H,
                W=W,
                device=output.device
            )

            filled = self.fill_strategy.fill(
                output[b:b + 1],
                mask
            )

            output[b:b + 1] = filled

        return [output, cover_image]