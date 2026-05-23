import torch
from .base import FillStrategy


class BlurFill(FillStrategy):
    """
    Use local context average (smoother than mean border).
    """

    def fill(self, image, b, top, left, h, w):

        patch = image[b, :, top-1:top+h+1, left-1:left+w+1]

        fill_value = patch.mean()

        image[b, :, top:top+h, left:left+w] = fill_value