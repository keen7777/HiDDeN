class FillStrategy:
    """
    Base class for all inpainting fill strategies.
    """

    def fill(self, image, b, top, left, h, w):
        """
        image: Tensor [B, C, H, W]
        b: batch index
        top, left: mask position
        h, w: mask size
        """
        raise NotImplementedError