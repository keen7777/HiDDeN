import torch
import torch.nn.functional as F
import numpy as np
from .base import FillStrategy

class PatchMatchFill:

    def __init__(self, patch_size=5, topk=200):
        self.patch_size = patch_size
        self.r = patch_size // 2
        self.topk = topk

    def fill(self, image, mask):
        """
        image: (B, C, H, W)
        mask:  (1, H, W) or (H, W)
        """

        B, C, H, W = image.shape
        output = image.clone()

        if mask.dim() == 3:
            mask = mask[0]
        mask = mask.bool()

        # -------------------------------------------------
        # STEP 1: extract ALL patches (GPU)
        # -------------------------------------------------
        patches = F.unfold(
            image,                    # (B,C,H,W)
            kernel_size=self.patch_size,
            padding=self.r
        )  # (B, C*P*P, N)

        patches = patches.permute(0, 2, 1)  # (B, N, D)

        N = patches.shape[1]

        mask_flat = mask.view(-1)
        mask_idx = torch.where(mask_flat)[0]
        known_idx = torch.where(~mask_flat)[0]

        if len(mask_idx) == 0 or len(known_idx) == 0:
            return output

        # -------------------------------------------------
        # STEP 2: sample known patches (IMPORTANT SPEED HACK)
        # -------------------------------------------------
        k = min(self.topk, len(known_idx))
        rand_idx = known_idx[torch.randperm(len(known_idx))[:k]]

        known_patches = patches[:, rand_idx, :]  # (B, k, D)

        # -------------------------------------------------
        # STEP 3: fill mask pixels (NO FULL SEARCH)
        # -------------------------------------------------
        for b in range(B):

            kp = known_patches[b]   # (k, D)
            img = output[b]

            for idx in mask_idx:

                p = patches[b, idx]  # (D,)

                # vectorized distance
                dist = torch.sum((kp - p) ** 2, dim=1)

                best = torch.argmin(dist)
                best_coord = rand_idx[best]

                # convert flat idx → (y,x)
                y = idx // W
                x = idx % W

                # copy center pixel of best patch
                output[b, :, y, x] = img.view(C, -1)[:, best_coord]

        return output