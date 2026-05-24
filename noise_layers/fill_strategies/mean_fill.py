# noise_layers/fill_strategies/mean_fill.py

import torch
from .base import FillStrategy


class MeanFill(FillStrategy):

    def fill(self, image, mask):

        # image: [1,C,H,W]
        B, C, H, W = image.shape

        mask = mask.bool()

        output = image.clone()

        for b in range(B):

            img = output[b]

            # pixels outside mask
            valid = ~mask

            for c in range(C):
                channel = img[c]

                mean_val = channel[valid].mean()

                channel[mask] = mean_val

                img[c] = channel

            output[b] = img

        return output