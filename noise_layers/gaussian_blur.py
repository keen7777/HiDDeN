import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def random_float(min, max):
    """
    Return a random float number in [min, max)
    """
    return np.random.rand() * (max - min) + min


class GaussianBlur(nn.Module):
    """
    Apply Gaussian Blur attack to the whole image.

    Main idea:
    - Randomly sample or manually set a blur strength sigma
    - Dynamically compute a suitable Gaussian kernel size
    - Construct a Gaussian kernel
    - Blur the image using depthwise convolution

    Gaussian Blur:
    - smooths the image spatially
    - removes local high-frequency details
    - weakens edges and texture information
    """

    def __init__(
        self,
        sigma=None,
        sigma_range=(0.5, 2.0)
    ):
        """
        :param sigma:
            Fixed sigma value.
            Useful for evaluation / sweep.
            Example:
                gaussianblur(1.5)

        :param sigma_range:
            Random sigma sampling range.
            Useful for training.
            Example:
                sigma_range=(0.5, 2.0)

        If sigma is not None:
            use fixed sigma

        If sigma is None:
            randomly sample sigma from sigma_range
        """

        super(GaussianBlur, self).__init__()

        self.sigma = sigma

        self.sigma_min = sigma_range[0]
        self.sigma_max = sigma_range[1]


    def get_kernel_size(self, sigma):
        """
        Dynamically compute kernel size from sigma.

        Rule:
            kernel_size = 2 * ceil(3 * sigma) + 1

        Reason:
            Gaussian values outside +-3 sigma are already very small.
            This kernel size covers most of the Gaussian distribution.

        Examples:
            sigma=0.5 -> kernel_size=5
            sigma=1.0 -> kernel_size=7
            sigma=1.5 -> kernel_size=11
            sigma=2.0 -> kernel_size=13

        :param sigma:
            Gaussian sigma

        :return:
            odd integer kernel size
        """

        kernel_size = int(2 * np.ceil(3 * sigma) + 1)

        # Safety check:
        # kernel_size should always be odd because of the formula above.
        if kernel_size % 2 == 0:
            kernel_size += 1

        return kernel_size


    def create_gaussian_kernel(self, sigma, kernel_size, device):
        """
        Create 2D Gaussian kernel.

        Gaussian formula:

            G(x, y) = exp(-(x^2 + y^2)/(2*sigma^2))

        Steps:
        1. Create coordinate grid
        2. Compute Gaussian values
        3. Normalize so kernel sums to 1

        :param sigma:
            Gaussian sigma

        :param kernel_size:
            Dynamically computed kernel size

        :param device:
            torch device

        :return:
            2D Gaussian kernel
        """

        k = kernel_size

        # Coordinate range:
        # Example:
        # kernel_size=7 -> [-3, -2, -1, 0, 1, 2, 3]
        coords = torch.arange(k, device=device).float() - k // 2

        # Create 2D coordinate grid.
        #
        # indexing='ij' is more explicit, but older PyTorch versions may not support it.
        # If your PyTorch supports it, you can use:
        # x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
        x_grid, y_grid = torch.meshgrid(coords, coords)

        # Convert sigma to tensor-like float operation.
        # sigma itself is a Python float, which is fine here.
        kernel = torch.exp(
            -(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2)
        )

        # Normalize the kernel.
        # This keeps image brightness roughly unchanged after blur.
        kernel = kernel / kernel.sum()

        return kernel


    def forward(self, noised_and_cover):
        """
        Apply Gaussian blur attack.

        noised_and_cover format:
            [0] -> encoded / noised image
            [1] -> original cover image

        We only modify the encoded image.
        """

        noised_image = noised_and_cover[0]

        # Training mode:
        # randomly sample sigma from sigma_range.
        #
        # Evaluation / sweep mode:
        # use fixed sigma.
        if self.sigma is None:
            sigma = random_float(self.sigma_min, self.sigma_max)
        else:
            sigma = self.sigma

        # Dynamically choose kernel size according to sigma.
        kernel_size = self.get_kernel_size(sigma)

        # Create Gaussian kernel.
        kernel = self.create_gaussian_kernel(
            sigma=sigma,
            kernel_size=kernel_size,
            device=noised_image.device
        )

        # Reshape kernel for depthwise convolution.
        #
        # F.conv2d expects kernel shape:
        #   (out_channels, in_channels / groups, kernel_height, kernel_width)
        #
        # For RGB image:
        #   noised_image.shape[1] = 3
        #
        # We want to apply the same Gaussian kernel independently to each channel.
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        kernel = kernel.repeat(
            noised_image.shape[1],  # number of image channels
            1,
            1,
            1
        )

        # Apply Gaussian blur.
        #
        # groups=noised_image.shape[1] means:
        #   each channel is convolved independently.
        #
        # padding=kernel_size // 2 means:
        #   output image keeps the same height and width.
        blurred_image = F.conv2d(
            noised_image,
            kernel,
            padding=kernel_size // 2,
            groups=noised_image.shape[1]
        )

        # Replace encoded image with blurred image.
        noised_and_cover[0] = blurred_image

        return noised_and_cover