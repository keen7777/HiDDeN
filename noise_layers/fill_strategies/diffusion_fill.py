import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FillStrategy


class DiffusionFill(FillStrategy):
    """
    Strict morphological inpainting:

    key idea:
    - ONLY known pixels contribute
    - boundary pixels are filled from known neighborhood
    - mask shrinks inward iteratively
    """

    def __init__(self, steps=10):
        self.steps = steps

        # 3x3 averaging kernel (no learning, pure propagation)
        self.kernel = torch.ones(1, 1, 3, 3)

    # =====================================================
    # compute boundary ring
    # =====================================================
    def get_boundary(self, mask):
        """
        boundary = mask - eroded(mask)
        """
        kernel = self.kernel.to(mask.device)

        # erode mask
        eroded = F.conv2d(mask, kernel, padding=1)
        eroded = (eroded == 9).float()

        boundary = mask - eroded
        return (boundary > 0).float()

    # =====================================================
    # propagate values from known region ONLY
    # =====================================================
    def propagate(self, x, known_mask):
        """
        x: image [B,3,H,W]
        known_mask: [B,1,H,W]
        """

        kernel = self.kernel.to(x.device)

        # expand kernel to RGB channels (IMPORTANT FIX)
        kernel_rgb = kernel.repeat(3, 1, 1, 1)

        # sum of known neighbors per channel
        sum_ = F.conv2d(x * known_mask, kernel_rgb, padding=1, groups=3)

        # count of known neighbors per channel
        count_ = F.conv2d(known_mask, kernel, padding=1, groups=1)

        # avoid division by zero
        avg = sum_ / (count_ + 1e-6)

        return avg

    # =====================================================
    # main fill function
    # =====================================================
    def fill(self, image, mask):
        """
        image: [B,3,H,W]
        mask : [H,W] (1 = missing, 0 = known)
        """

        B, C, H, W = image.shape
        device = image.device

        # -------------------------
        # prepare mask
        # -------------------------
        mask = mask.float().to(device)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

        known_mask = 1.0 - mask

        # -------------------------
        # initialization
        # -------------------------
        x = image.clone()

        # unknown region initialized as noise (optional but stable)
        x = x * known_mask + torch.randn_like(x) * mask

        current_mask = mask.clone()

        # =================================================
        # inward propagation loop
        # =================================================
        for _ in range(self.steps):

            # 1. compute boundary of missing region
            boundary = self.get_boundary(current_mask)

            if boundary.sum() == 0:
                break

            # 2. compute values from known region ONLY
            propagated = self.propagate(x, known_mask)

            # 3. update ONLY boundary pixels
            x = x * (1 - boundary) + propagated * boundary

            # 4. shrink mask (move inward)
            current_mask = current_mask - boundary
            current_mask = (current_mask > 0).float()

            # 5. enforce hard constraint (VERY IMPORTANT)
            x = x * known_mask + image * mask

        return x