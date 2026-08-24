import numpy as np
import torch
import torch.nn as nn

from noise_layers.mask_generator import RectangleRemovalMaskGenerator


# ============================================================
# Basic U-Net building block
# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Small U-Net reconstruction model
# ============================================================

class SmallUNet(nn.Module):
    """
    Small U-Net for masked image reconstruction.

    Input:
        RGB masked image : 3 channels
        binary mask      : 1 channel

    Total:
        4 input channels

    Image range:
        [-1, 1]

    mask:
        1 = missing / reconstruct
        0 = known / retain
    """

    def __init__(self, base_channels=32):
        super().__init__()

        c1 = base_channels          # 32
        c2 = base_channels * 2      # 64
        c3 = base_channels * 4      # 128
        c4 = base_channels * 8      # 256

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.enc1 = ConvBlock(
            4,
            c1,
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(
            c1,
            c2,
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(
            c2,
            c3,
        )

        self.pool3 = nn.MaxPool2d(2)

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = ConvBlock(
            c3,
            c4,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up3 = nn.ConvTranspose2d(
            c4,
            c3,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = ConvBlock(
            c3 + c3,
            c3,
        )

        self.up2 = nn.ConvTranspose2d(
            c3,
            c2,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = ConvBlock(
            c2 + c2,
            c2,
        )

        self.up1 = nn.ConvTranspose2d(
            c2,
            c1,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = ConvBlock(
            c1 + c1,
            c1,
        )

        # ----------------------------------------------------
        # RGB prediction
        # ----------------------------------------------------

        self.output = nn.Sequential(
            nn.Conv2d(
                c1,
                3,
                kernel_size=1,
            ),
            nn.Tanh(),
        )

    def forward(self, image, mask):
        """
        image:
            [B, 3, H, W], range [-1, 1]

        mask:
            [B, 1, H, W]
            1 = missing
            0 = known
        """

        # ----------------------------------------------------
        # Remove masked pixels
        # ----------------------------------------------------

        masked_image = (
            image * (1.0 - mask)
        )

        x = torch.cat(
            [masked_image, mask],
            dim=1,
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        b = self.bottleneck(
            self.pool3(e3)
        )

        # ----------------------------------------------------
        # Decoder + skip connections
        # ----------------------------------------------------

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )

        d1 = self.dec1(d1)

        prediction = self.output(d1)

        # ----------------------------------------------------
        # Strict blending:
        #
        # known region:
        #   always original image
        #
        # missing region:
        #   U-Net prediction
        # ----------------------------------------------------

        reconstructed = (
            image * (1.0 - mask)
            + prediction * mask
        )

        return reconstructed


# ============================================================
# Eval-aligned rectangular removal mask generator
# ============================================================

class ControlledRectangleMaskGenerator:
    """
    Ratio-sampling wrapper around the EXACT rectangle-mask generator
    used by evaluation.

    Spatial mask geometry is delegated to
    RectangleRemovalMaskGenerator from noise_layers.mask_generator.

    Mask convention:
        1 = removed / missing / reconstructed
        0 = retained / known

    Important:
        - Same rectangle geometry as EvalInpainting.
        - Same defaults as evaluation:
              min_mask_size=8
              max_aspect_ratio=3.0
              max_rectangles=50
              max_rectangle_ratio=0.10
              margin=1
        - No isolated single-pixel fallback.
        - Actual removal ratio may be slightly below the requested one.
        - Removal-ratio RNG and mask-geometry RNG are kept separate,
          exactly like EvalInpainting.
    """

    def __init__(
        self,
        min_ratio=0.1,
        max_ratio=0.5,
        min_mask_size=8,
        max_aspect_ratio=3.0,
        max_rectangles=50,
        max_rectangle_ratio=0.10,
        margin=1,
        seed=42,
        randomize_ratio=True,
    ):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.randomize_ratio = randomize_ratio

        if not 0.0 <= self.min_ratio <= 1.0:
            raise ValueError(
                f"min_ratio must be in [0, 1], got {self.min_ratio}"
            )

        if not 0.0 <= self.max_ratio <= 1.0:
            raise ValueError(
                f"max_ratio must be in [0, 1], got {self.max_ratio}"
            )

        if self.min_ratio > self.max_ratio:
            raise ValueError(
                f"min_ratio ({self.min_ratio}) must be <= "
                f"max_ratio ({self.max_ratio})"
            )

        # Same design as EvalInpainting:
        # one RNG samples the severity, another RNG samples geometry.
        self.ratio_rng = np.random.RandomState(seed)

        mask_seed = (
            None if seed is None
            else seed + 1
        )

        self.rectangle_generator = RectangleRemovalMaskGenerator(
            min_mask_size=min_mask_size,
            max_aspect_ratio=max_aspect_ratio,
            max_rectangles=max_rectangles,
            max_rectangle_ratio=max_rectangle_ratio,
            margin=margin,
            seed=mask_seed,
        )

        self.last_removal_ratios = None

    def _sample_removal_ratio(self):
        if self.randomize_ratio:
            return float(
                self.ratio_rng.uniform(
                    self.min_ratio,
                    self.max_ratio,
                )
            )

        return float(self.max_ratio)

    def generate_one(
        self,
        H,
        W,
        removal_ratio,
    ):
        """
        Generate one mask using the exact evaluation geometry.
        """
        return self.rectangle_generator.generate_one(
            H=H,
            W=W,
            removal_ratio=removal_ratio,
        )

    def _generate_one(
        self,
        H,
        W,
    ):
        removal_ratio = self._sample_removal_ratio()

        mask = self.generate_one(
            H=H,
            W=W,
            removal_ratio=removal_ratio,
        )

        return mask, removal_ratio

    def generate(
        self,
        B,
        H,
        W,
        device,
    ):
        """
        Batch API kept compatible with the existing pretraining and
        FrozenPretrainedUNetInpainting code.

        Returns:
            [B, 1, H, W] float tensor
        """
        masks = []
        removal_ratios = []

        for _ in range(B):
            mask, removal_ratio = self._generate_one(
                H=H,
                W=W,
            )

            masks.append(mask)
            removal_ratios.append(removal_ratio)

        masks = np.stack(
            masks,
            axis=0,
        )

        masks = torch.tensor(
            masks,
            dtype=torch.float32,
            device=device,
        )

        self.last_removal_ratios = removal_ratios

        return masks.unsqueeze(1)
