import torch
import torch.nn as nn
import numpy as np
import cv2


class MaskInpaintingTelea(nn.Module):
    """
    ==========================
    Clean Telea Inpainting
    ==========================

    Design goals:
    1. strictly reproducible mask
    2. correct OpenCV pipeline
    3. no artificial post-processing
    4. correct normalization handling

    Output space:
        -> matches model space [-1, 1]
    """

    def __init__(self,
                 mask_size_range_min,
                 mask_size_range_max,
                 seed=42):

        super(MaskInpaintingTelea, self).__init__()

        self.mask_min = mask_size_range_min
        self.mask_max = mask_size_range_max

        # deterministic randomness (important for sweep reproducibility)
        self.rng = np.random.RandomState(seed)


    def forward(self, noised_and_cover, debug=False, strength=None):

        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        _, _, H, W = noised_image.shape
        output_image = noised_image.clone()

        # ======================================================
        # 1. sample mask (fixed per forward call)
        # ======================================================
        mask_ratio = self.rng.uniform(self.mask_min, self.mask_max)

        mask_h = max(1, int(H * mask_ratio))
        mask_w = max(1, int(W * mask_ratio))

        top = self.rng.randint(1, H - mask_h - 1)
        left = self.rng.randint(1, W - mask_w - 1)

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[top:top + mask_h, left:left + mask_w] = 255


        # ======================================================
        # 2. per-image processing
        # ======================================================
        for i in range(noised_image.shape[0]):

            img = output_image[i].detach()

            # --------------------------------------------------
            # A. unify input space → [0,1]
            # --------------------------------------------------
            if img.min() < 0:          # assume [-1,1]
                img = (img + 1) / 2.0

            img = torch.clamp(img, 0.0, 1.0)

            # --------------------------------------------------
            # B. tensor → numpy (HWC uint8)
            # --------------------------------------------------
            img = img.permute(1, 2, 0).cpu().numpy()
            img_uint8 = (img * 255.0).round().astype(np.uint8)

            # --------------------------------------------------
            # C. OpenCV Telea inpainting
            # --------------------------------------------------
            telea = cv2.inpaint(
                img_uint8,
                mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )

            # --------------------------------------------------
            # D. back to float [0,1]
            # --------------------------------------------------
            result = torch.from_numpy(telea).permute(2, 0, 1).float() / 255.0

            # --------------------------------------------------
            # E. CRITICAL: convert back to model space [-1,1]
            # --------------------------------------------------
            result = result * 2.0 - 1.0

            output_image[i] = result.to(output_image.device)

        return [output_image, cover_image]