import torch
import torch.nn.functional as F
from .base import FillStrategy


class BlurFill(FillStrategy):

    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def fill(self, image, mask):

        B, C, H, W = image.shape

        # =========================
        # correct depthwise kernel
        # =========================
        kernel = torch.ones((C, 1, 3, 3), device=image.device) / 9.0

        blurred = F.conv2d(
            image,
            kernel,
            padding=1,
            groups=C
        )

        mask = mask.unsqueeze(0).unsqueeze(0).float()

        return image * (1 - mask) + blurred * mask