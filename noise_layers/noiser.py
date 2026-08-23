import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.frozen_pretrained_unet_inpainting import (
    FrozenPretrainedUNetInpainting,
)


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        self.noise_layers = nn.ModuleList()
        self.noise_layers.append(Identity())
        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))
                elif layer == 'QuantizationPlaceholder':
                    self.noise_layers.append(Quantization(device))
                elif layer.startswith('FrozenUNetPlaceholder|'):
                    parts = layer.split('|')

                    if len(parts) != 5:
                        raise ValueError(
                            f'Invalid FrozenUNetPlaceholder: {layer}'
                        )

                    _, checkpoint_path, min_ratio, max_ratio, seed = parts

                    frozen_unet = FrozenPretrainedUNetInpainting(
                        checkpoint_path=checkpoint_path,
                        min_ratio=float(min_ratio),
                        max_ratio=float(max_ratio),
                        seed=int(seed),
                    )

                    self.noise_layers.append(
                        frozen_unet
                    )
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
                                     f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, encoded_and_cover):
        idx = np.random.randint(len(self.noise_layers))
        random_noise_layer = self.noise_layers[idx]
        return random_noise_layer(encoded_and_cover)

