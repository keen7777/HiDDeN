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


class MaskInpainting(nn.Module):
    """
    Random multi-mask inpainting attack.

    Main idea:
    1. Randomly choose a TOTAL mask ratio
    2. Randomly choose number of masks
    3. Split total mask area into several smaller masks
    4. Randomly generate rectangle masks with random aspect ratios
    5. Fill masked regions using neighboring pixels

    Compared with the old version:
    - supports multiple masks
    - supports random aspect ratio
    - supports random total area
    - each image has independent masks
    - more realistic corruption distribution
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
        super(MaskInpainting, self).__init__()

        # maximum TOTAL masked area ratio
        # e.g. 0.5 means at most 50% of the image is masked
        self.max_mask_ratio = max_mask_ratio

        # maximum number of masks per image
        self.max_mask_number = max_mask_number

        # minimum mask side length
        self.min_mask_size = min_mask_size

        # maximum aspect ratio
        # 3.0 means:
        # width/height can vary between:
        # 1/3 and 3
        self.max_aspect_ratio = max_aspect_ratio

        # choose a strategy
        self.fill_strategy = fill_strategy

        # random generator
        # if seed is None -> fully random
        self.rng = np.random.RandomState(seed)
        # use random while training, turn off while sweep
        self.randomize_ratio = randomize_ratio

    # 1. Generate random masks   
    def generate_random_masks(self, H, W):

        masks = []

        # 🎯 target TRUE coverage
        if self.randomize_ratio:
            mask_ratio = self.rng.uniform(
                0.05,
                self.max_mask_ratio
            )
        else:
            mask_ratio = self.max_mask_ratio
        target_coverage = mask_ratio

        # pixel-level union mask
        mask_union = np.zeros((H, W), dtype=np.uint8)

        max_w = W - 4
        max_h = H - 4

        iters = 0
        max_iters = 300

        while mask_union.mean() < target_coverage and iters < max_iters:

            iters += 1

            aspect_ratio = self.rng.uniform(
                1 / self.max_aspect_ratio,
                self.max_aspect_ratio
            )

            # sample candidate area (rough control)
            area = self.rng.uniform(
                self.min_mask_size ** 2,
                H * W * 0.1  # optional cap
            )

            w = int(np.sqrt(area * aspect_ratio))
            h = int(np.sqrt(area / aspect_ratio))

            w = max(1, min(w, max_w))
            h = max(1, min(h, max_h))

            if max_w - w <= 1 or max_h - h <= 1:
                continue

            left = self.rng.randint(1, max_w - w)
            top = self.rng.randint(1, max_h - h)

            # apply to union mask
            mask_union[top:top+h, left:left+w] = 1

            masks.append((top, left, h, w))

        return masks
    
    
    # Forward
    def forward(self, noised_and_cover):
        noised_image, cover_image = noised_and_cover

        # noised_image = noised_image.detach()
        B, C, H, W = noised_image.shape

        output = noised_image.clone()

        for b in range(B):

            mask = torch.zeros((H, W), device=output.device)

            masks = self.generate_random_masks(H, W)

            for top, left, h, w in masks:
                mask[top:top+h, left:left+w] = 1.0

            output[b] = self.fill_strategy.fill(
                output[b:b+1],
                mask
            )

        return [output, cover_image]