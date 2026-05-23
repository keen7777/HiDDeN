# noise_layers/fill_strategies/telea_fill.py

import torch
import numpy as np
import cv2
from .base import FillStrategy


class TeleaFill(FillStrategy):

    def __init__(self, radius=3):
        self.radius = radius

    def fill(self, image, mask, b):

        img = image[b].detach()

        # [-1,1] → [0,1]
        if img.min() < 0:
            img = (img + 1) / 2.0

        img = torch.clamp(img, 0.0, 1.0)

        img = img.permute(1, 2, 0).cpu().numpy()
        img_uint8 = (img * 255.0).astype(np.uint8)

        # OpenCV expects single-channel or 3-channel mask
        telea = cv2.inpaint(
            img_uint8,
            mask,
            self.radius,
            cv2.INPAINT_TELEA
        )

        result = torch.from_numpy(telea).permute(2, 0, 1).float() / 255.0
        result = result * 2.0 - 1.0

        image[b] = result.to(image.device)