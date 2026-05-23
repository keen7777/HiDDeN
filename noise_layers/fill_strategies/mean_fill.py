import torch
from .base import FillStrategy


class MeanFill(FillStrategy):
    """
    Fill masked region with mean of surrounding pixels.
    """

    def fill(self, image, b, top, left, h, w):

        neighbors = []

        # top
        neighbors.append(image[b, :, top - 1, left:left + w])

        # bottom
        neighbors.append(image[b, :, top + h, left:left + w])

        # left
        neighbors.append(image[b, :, top:top + h, left - 1])

        # right
        neighbors.append(image[b, :, top:top + h, left + w])

        neighbors = torch.cat([x.reshape(-1) for x in neighbors])

        fill_value = neighbors.mean()

        image[b, :, top:top + h, left:left + w] = fill_value