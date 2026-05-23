import torch
from .base import FillStrategy


class RandomNeighborFill(FillStrategy):
    """
    Fill with a random pixel from boundary.
    """

    def fill(self, image, b, top, left, h, w):

        neighbors = []

        neighbors.append(image[b, :, top - 1, left:left + w])
        neighbors.append(image[b, :, top + h, left:left + w])
        neighbors.append(image[b, :, top:top + h, left - 1])
        neighbors.append(image[b, :, top:top + h, left + w])

        neighbors = torch.cat([x.reshape(-1) for x in neighbors])

        idx = torch.randint(0, neighbors.shape[0], (1,))
        fill_value = neighbors[idx]

        image[b, :, top:top + h, left:left + w] = fill_value