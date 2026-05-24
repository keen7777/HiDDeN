import torch
from .base import FillStrategy
class RandomNeighborFill(FillStrategy):

    def fill(self, image, mask):

        B, C, H, W = image.shape
        mask = mask.bool()

        output = image.clone()

        for b in range(B):
            img = output[b]

            valid_pixels = img[:, ~mask].reshape(C, -1)

            for c in range(C):
                sampled = valid_pixels[c][
                    torch.randint(0, valid_pixels.shape[1], (mask.sum(),))
                ]

                img[c][mask] = sampled

            output[b] = img

        return output