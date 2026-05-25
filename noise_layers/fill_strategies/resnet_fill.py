# noise_layers/fill_strategies/resnet_fill.py

import torch
import torch.nn as nn

from .base import FillStrategy


# =========================================================
# 1. Standard ResBlock (NO mask dependency)
# =========================================================
class ResBlock(nn.Module):
    """
    回退到原始版本：
    - 保证 checkpoint 完全兼容
    - 不引入 mask-aware forward（否则无法 load）
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


# =========================================================
# 2. Compatible ResNetFill (4-channel version)
# =========================================================
class ResNetFill(FillStrategy, nn.Module):

    def __init__(self, hidden_channels=32, num_blocks=4):
        """
        ⚠️ 必须和旧 checkpoint 完全一致
        """

        FillStrategy.__init__(self)
        nn.Module.__init__(self)

        # -------------------------------------------------
        # INPUT: EXACTLY 4 channels
        #   RGB (3)
        #   mask (1)
        # -------------------------------------------------
        self.net = nn.Sequential(

            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            *[ResBlock(hidden_channels) for _ in range(num_blocks)],

            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    # =====================================================
    # forward fill
    # =====================================================
    def fill(self, image, mask):
        """
        image: [B,3,H,W]
        mask : [H,W] (1 = missing, 0 = valid)
        """

        B, C, H, W = image.shape

        # -------------------------------------------------
        # mask -> [B,1,H,W]
        # -------------------------------------------------
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        mask = mask.expand(B, 1, H, W)

        # -------------------------------------------------
        # masked image
        # -------------------------------------------------
        masked_image = image * (1 - mask)

        # -------------------------------------------------
        # concat EXACTLY 4 channels
        # -------------------------------------------------
        inp = torch.cat([masked_image, mask], dim=1)

        # -------------------------------------------------
        # forward
        # -------------------------------------------------
        reconstructed = self.net(inp)

        # -------------------------------------------------
        # strict blending (same as original logic)
        # -------------------------------------------------
        output = image * (1 - mask) + reconstructed * mask

        return output