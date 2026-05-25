import cv2
import numpy as np
import torch
from .base import FillStrategy


class TeleaFill(FillStrategy):

    def __init__(self, radius=3):
        self.radius = radius

    def fill(self, image, mask):

        # image: [1,C,H,W]
        img = image[0]

        if img.min() < 0:
            img = (img + 1) / 2.0

        img = torch.clamp(img, 0, 1)

        img = img.detach().permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)

        out = cv2.inpaint(img, mask_np, self.radius, cv2.INPAINT_TELEA)

        out = torch.from_numpy(out).float() / 255.0
        out = out.permute(2,0,1)

        out = out * 2 - 1

        return out.unsqueeze(0).to(image.device)