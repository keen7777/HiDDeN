import numpy as np
import torch
import torch.nn as nn


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
# Controlled rectangular union mask generator
# ============================================================

class ControlledRectangleMaskGenerator:
    """
    Generate union-of-rectangle masks with controlled coverage.

    Unlike the old implementation, complete rectangles are not
    allowed to overshoot the target pixel count.

    The generator keeps adding rectangles whose maximum area is
    bounded by the number of remaining pixels.

    Result:
        actual coverage ~= requested target coverage

    mask:
        1 = missing
        0 = known
    """

    def __init__(
        self,
        min_ratio=0.1,
        max_ratio=0.5,
        min_mask_size=8,
        max_aspect_ratio=4.0,
        seed=42,
        randomize_ratio=True,
    ):

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        self.min_mask_size = min_mask_size
        self.max_aspect_ratio = max_aspect_ratio

        self.randomize_ratio = randomize_ratio

        self.rng = np.random.RandomState(
            seed
        )

    # --------------------------------------------------------
    # Sample rectangle dimensions
    # --------------------------------------------------------

    def _sample_rectangle_size(
        self,
        max_area,
        H,
        W,
    ):

        if max_area <= 0:
            return 1, 1

        min_area = (
            self.min_mask_size ** 2
        )

        # Near the target, allow the final rectangle
        # to become smaller than min_mask_size.
        if max_area <= min_area:
            area = max_area

        else:
            area = self.rng.uniform(
                min_area,
                max_area,
            )

        aspect_ratio = self.rng.uniform(
            1.0 / self.max_aspect_ratio,
            self.max_aspect_ratio,
        )

        w = max(
            1,
            int(
                np.sqrt(
                    area * aspect_ratio
                )
            ),
        )

        h = max(
            1,
            int(
                np.sqrt(
                    area / aspect_ratio
                )
            ),
        )

        w = min(w, W)
        h = min(h, H)

        # Integer rounding may occasionally create
        # an area slightly larger than max_area.
        while (
            w * h > max_area
            and (w > 1 or h > 1)
        ):

            if w >= h and w > 1:
                w -= 1

            elif h > 1:
                h -= 1

            else:
                break

        return h, w

    # --------------------------------------------------------
    # Generate one mask
    # --------------------------------------------------------

    def _generate_one(
        self,
        H,
        W,
    ):

        if self.randomize_ratio:

            target_ratio = self.rng.uniform(
                self.min_ratio,
                self.max_ratio,
            )

        else:

            target_ratio = self.max_ratio

        total_pixels = H * W

        target_pixels = int(
            round(
                target_ratio
                * total_pixels
            )
        )

        target_pixels = max(
            1,
            min(
                target_pixels,
                total_pixels,
            ),
        )

        mask = np.zeros(
            (H, W),
            dtype=np.float32,
        )

        current_pixels = 0

        max_iterations = 2000

        iterations = 0

        while (
            current_pixels < target_pixels
            and iterations < max_iterations
        ):

            iterations += 1

            remaining = (
                target_pixels
                - current_pixels
            )

            # Keep the old experiment's basic idea:
            # one candidate rectangle should not dominate
            # more than 10% of the full image.
            max_rectangle_area = min(
                int(
                    total_pixels * 0.1
                ),
                remaining,
            )

            max_rectangle_area = max(
                1,
                max_rectangle_area,
            )

            h, w = (
                self._sample_rectangle_size(
                    max_rectangle_area,
                    H,
                    W,
                )
            )

            # ------------------------------------------------
            # Try several positions and choose the position
            # adding the most NEW pixels.
            #
            # This reduces excessive overlap.
            # ------------------------------------------------

            best_top = None
            best_left = None
            best_new_pixels = -1

            trials = 20

            for _ in range(trials):

                if H == h:
                    top = 0
                else:
                    top = self.rng.randint(
                        0,
                        H - h + 1,
                    )

                if W == w:
                    left = 0
                else:
                    left = self.rng.randint(
                        0,
                        W - w + 1,
                    )

                region = mask[
                    top:top + h,
                    left:left + w
                ]

                new_pixels = int(
                    np.sum(
                        region == 0
                    )
                )

                if (
                    new_pixels
                    > best_new_pixels
                ):

                    best_new_pixels = (
                        new_pixels
                    )

                    best_top = top
                    best_left = left

            if (
                best_top is None
                or best_new_pixels <= 0
            ):
                continue

            # ------------------------------------------------
            # Since rectangle area <= remaining,
            # this cannot overshoot target_pixels.
            # ------------------------------------------------

            mask[
                best_top:best_top + h,
                best_left:best_left + w
            ] = 1.0

            current_pixels = int(
                mask.sum()
            )

        # ----------------------------------------------------
        # Safety fallback:
        #
        # If highly overlapping rectangles prevent exact
        # coverage, fill remaining pixels individually.
        #
        # This should normally affect only a tiny number
        # of final pixels.
        # ----------------------------------------------------

        remaining = (
            target_pixels
            - int(mask.sum())
        )

        if remaining > 0:

            empty = np.argwhere(
                mask == 0
            )

            chosen = self.rng.choice(
                len(empty),
                size=min(
                    remaining,
                    len(empty),
                ),
                replace=False,
            )

            coords = empty[chosen]

            mask[
                coords[:, 0],
                coords[:, 1]
            ] = 1.0

        return mask

    # --------------------------------------------------------
    # Batch API
    # --------------------------------------------------------

    def generate(
        self,
        B,
        H,
        W,
        device,
    ):

        masks = []

        for _ in range(B):

            masks.append(
                self._generate_one(
                    H,
                    W,
                )
            )

        mask = np.stack(
            masks,
            axis=0,
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32,
            device=device,
        )

        return mask.unsqueeze(1)