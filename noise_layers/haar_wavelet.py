import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWavelet(nn.Module):
    """
    Differentiable Haar wavelet noise layer.

    Main idea:
    1. Decompose the encoded image into four Haar wavelet sub-bands:
       LL, LH, HL, HH.
    2. Apply perturbation to selected sub-bands, usually high-frequency bands:
       LH, HL, HH.
    3. Reconstruct the attacked image using inverse Haar wavelet transform.

    This layer is designed to be differentiable and can be used during training.

    Parameters:
        strength:
            Attack strength in [0, 1].
            For mode="attenuate", strength=0.3 means high-frequency bands are
            multiplied by 0.7.

        mode:
            "attenuate": weaken selected wavelet bands.
            "dropout": randomly remove coefficients in selected bands.
            "soft_threshold": suppress small coefficients.

        attack_bands:
            Tuple/list of bands to attack.
            Usually ("LH", "HL", "HH").
            You can also test ("LL",) or ("LL", "LH", "HL", "HH"), but this is
            usually much more destructive.

        clamp:
            Whether to clamp reconstructed image to [0, 1].
    """

    def __init__(
        self,
        strength=0.3,
        mode="attenuate",
        attack_bands=("LH", "HL", "HH"),
        clamp=True
    ):
        super(HaarWavelet, self).__init__()

        assert 0.0 <= strength <= 1.0, "strength should be in [0, 1]"
        assert mode in ["attenuate", "dropout", "soft_threshold"], \
            "mode should be one of: attenuate, dropout, soft_threshold"

        valid_bands = {"LL", "LH", "HL", "HH"}
        for band in attack_bands:
            assert band in valid_bands, "attack_bands can only contain LL, LH, HL, HH"

        self.strength = strength
        self.mode = mode
        self.attack_bands = attack_bands
        self.clamp = clamp

    def haar_dwt(self, image):
        """
        Haar discrete wavelet transform.

        Input:
            image: [B, C, H, W]

        Output:
            LL, LH, HL, HH: each has shape [B, C, H/2, W/2]
        """

        # If H or W is odd, remove the last row/column.
        # Most HiDDeN images are usually square and even-sized, e.g. 128x128.
        if image.shape[2] % 2 != 0:
            image = image[:, :, :-1, :]
        if image.shape[3] % 2 != 0:
            image = image[:, :, :, :-1]

        x00 = image[:, :, 0::2, 0::2]
        x01 = image[:, :, 0::2, 1::2]
        x10 = image[:, :, 1::2, 0::2]
        x11 = image[:, :, 1::2, 1::2]

        # Haar decomposition.
        # The factor 2.0 keeps the inverse transform consistent.
        LL = (x00 + x01 + x10 + x11) / 2.0
        LH = (x00 - x01 + x10 - x11) / 2.0
        HL = (x00 + x01 - x10 - x11) / 2.0
        HH = (x00 - x01 - x10 + x11) / 2.0

        return LL, LH, HL, HH

    def haar_idwt(self, LL, LH, HL, HH):
        """
        Inverse Haar discrete wavelet transform.

        Input:
            LL, LH, HL, HH: [B, C, H, W]

        Output:
            reconstructed image: [B, C, 2H, 2W]
        """

        x00 = (LL + LH + HL + HH) / 2.0
        x01 = (LL - LH + HL - HH) / 2.0
        x10 = (LL + LH - HL - HH) / 2.0
        x11 = (LL - LH - HL + HH) / 2.0

        batch_size, channels, height, width = LL.shape

        image = torch.zeros(
            batch_size,
            channels,
            height * 2,
            width * 2,
            device=LL.device,
            dtype=LL.dtype
        )

        image[:, :, 0::2, 0::2] = x00
        image[:, :, 0::2, 1::2] = x01
        image[:, :, 1::2, 0::2] = x10
        image[:, :, 1::2, 1::2] = x11

        return image

    def attack_band(self, band):
        """
        Apply perturbation to one wavelet sub-band.
        """

        if self.mode == "attenuate":
            # Simple differentiable high-frequency weakening.
            return band * (1.0 - self.strength)

        elif self.mode == "dropout":
            # Training-compatible random coefficient removal.
            # The random mask itself is not learnable, but gradients still pass
            # through the remaining coefficients.
            mask = (torch.rand_like(band) > self.strength).float()
            return band * mask

        elif self.mode == "soft_threshold":
            # Suppress small wavelet coefficients.
            # This is similar to a denoising-style operation.
            threshold = self.strength * band.abs().mean(dim=(2, 3), keepdim=True)
            return torch.sign(band) * F.relu(torch.abs(band) - threshold)

        else:
            return band

    def forward(self, noised_and_cover):
        """
        Compatible with the current HiDDeN-style noise layer interface.

        Input:
            noised_and_cover[0]: encoded/noised image
            noised_and_cover[1]: cover image

        Output:
            noised_and_cover with noised_and_cover[0] replaced by wavelet-attacked image.
        """

        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        assert noised_image.shape == cover_image.shape

        original_height = noised_image.shape[2]
        original_width = noised_image.shape[3]

        LL, LH, HL, HH = self.haar_dwt(noised_image)

        if "LL" in self.attack_bands:
            LL = self.attack_band(LL)
        if "LH" in self.attack_bands:
            LH = self.attack_band(LH)
        if "HL" in self.attack_bands:
            HL = self.attack_band(HL)
        if "HH" in self.attack_bands:
            HH = self.attack_band(HH)

        attacked_image = self.haar_idwt(LL, LH, HL, HH)

        # Restore original size in case H or W was odd.
        attacked_image = attacked_image[:, :, :original_height, :original_width]

        if self.clamp:
            attacked_image = torch.clamp(attacked_image, 0.0, 1.0)

        noised_and_cover[0] = attacked_image

        return noised_and_cover