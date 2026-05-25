import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. Stochastic Noise Injection
# =========================================================
class NoiseInjection(nn.Module):
    """
    inject randomness so generator is NOT deterministic
    """

    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn_like(x)
        return x + self.scale * noise


# =========================================================
# 2. Small Residual Block
# =========================================================
class ResBlock(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1)
        )

    def forward(self, x):
        return x + self.net(x)


# =========================================================
# 3. Corruption Generator
# =========================================================
class CorruptionGenerator(nn.Module):

    def __init__(self, in_channels=4, hidden=32, num_blocks=3):
        """
        in_channels = 3 (image) + 1 (mask)
        """

        super().__init__()

        # ----------------------------
        # Encoder
        # ----------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # ----------------------------
        # stochastic bottleneck
        # ----------------------------
        self.noise = NoiseInjection(hidden)

        # ----------------------------
        # middle residual blocks
        # ----------------------------
        self.blocks = nn.ModuleList([
            ResBlock(hidden) for _ in range(num_blocks)
        ])

        # ----------------------------
        # Decoder
        # ----------------------------
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1),
            nn.Sigmoid()
        )

    # =====================================================
    # forward
    # =====================================================
    def forward(self, image, mask):

        """
        image: [B,3,H,W]
        mask : [B,1,H,W]
        """

        # --------------------------------
        # masked input
        # --------------------------------
        masked_image = image * (1 - mask)

        x = torch.cat([masked_image, mask], dim=1)

        # --------------------------------
        # encode
        # --------------------------------
        x = self.encoder(x)

        # --------------------------------
        # stochasticity injection
        # --------------------------------
        x = self.noise(x)

        # --------------------------------
        # residual processing
        # --------------------------------
        for blk in self.blocks:
            x = blk(x)

        # --------------------------------
        # decode corruption
        # --------------------------------
        corruption = self.decoder(x)

        # --------------------------------
        # final blend
        # only affect mask region
        # --------------------------------
        out = image * (1 - mask) + corruption * mask

        return out