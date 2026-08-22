import numpy as np
import torch
import torch.nn as nn


class ReconstructionCNN(nn.Module):
    """
    Small CNN for masked image reconstruction.

    Architecture intentionally follows the original CorrNet:
        4 -> 32 -> 32 -> 32 -> 3

    Image range:
        [-1, 1]
    """

    def __init__(self, hidden_channels=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, 3, 3, padding=1),

            # HiDDeN images are normalized to [-1, 1]
            nn.Tanh()
        )

    def forward(self, image, mask):
        """
        image: [B, 3, H, W]
        mask : [B, 1, H, W]
               1 = missing
               0 = known
        """

        masked_image = image * (1.0 - mask)

        inp = torch.cat(
            [masked_image, mask],
            dim=1
        )

        prediction = self.net(inp)

        # Keep known pixels unchanged.
        output = (
            image * (1.0 - mask)
            + prediction * mask
        )

        return output


class RectangleMaskGenerator:
    """
    Multi-rectangle union masks matching the basic mask geometry
    of the original CNN corruption experiment.

    mask = 1 -> missing
    mask = 0 -> known
    """

    def __init__(
        self,
        min_ratio=0.1,
        max_ratio=0.5,
        min_mask_size=8,
        max_aspect_ratio=4.0,
        seed=42,
    ):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_mask_size = min_mask_size
        self.max_aspect_ratio = max_aspect_ratio

        self.rng = np.random.RandomState(seed)

    def generate(self, B, H, W, device):

        batch_masks = []

        for _ in range(B):

            target_coverage = self.rng.uniform(
                self.min_ratio,
                self.max_ratio
            )

            mask_union = np.zeros(
                (H, W),
                dtype=np.float32
            )

            max_w = W - 4
            max_h = H - 4

            iterations = 0
            max_iterations = 300

            while (
                mask_union.mean() < target_coverage
                and iterations < max_iterations
            ):

                iterations += 1

                aspect_ratio = self.rng.uniform(
                    1.0 / self.max_aspect_ratio,
                    self.max_aspect_ratio
                )

                area = self.rng.uniform(
                    self.min_mask_size ** 2,
                    H * W * 0.1
                )

                w = int(np.sqrt(area * aspect_ratio))
                h = int(np.sqrt(area / aspect_ratio))

                w = max(1, min(w, max_w))
                h = max(1, min(h, max_h))

                if max_w - w <= 1 or max_h - h <= 1:
                    continue

                left = self.rng.randint(
                    1,
                    max_w - w
                )

                top = self.rng.randint(
                    1,
                    max_h - h
                )

                mask_union[
                    top:top + h,
                    left:left + w
                ] = 1.0

            batch_masks.append(mask_union)

        mask = np.stack(batch_masks)

        mask = torch.tensor(
            mask,
            dtype=torch.float32,
            device=device
        )

        return mask.unsqueeze(1)