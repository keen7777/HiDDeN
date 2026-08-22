import argparse
import re
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.identity import Identity
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.quantization import Quantization
from noise_layers.jpeg_compression import JpegCompression
# adding new attacking methods:

# from archive.mask_inpainting_telea import MaskInpaintingTelea
from noise_layers.haar_wavelet import HaarWavelet
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.fixed_cnn_inpainting import FixedCNNInpainting

# maybe need to delete resnet and diffusion
from noise_layers.eval_inpainting import EvalInpainting
from noise_layers.fill_strategies.mean_fill import MeanFill
from noise_layers.fill_strategies.random_fill import RandomNeighborFill
# add naive stoke, and other evaluate method??
from noise_layers.fill_strategies.blur_fill import BlurFill
from noise_layers.fill_strategies.telea_fill import TeleaFill
from noise_layers.fill_strategies.patchmatch_fill import PatchMatchFill
# need pre-trained model? and other parameters???
from noise_layers.fill_strategies.resnet_fill import ResNetFill
from noise_layers.fill_strategies.diffusion_fill import DiffusionFill


# map string -> class
FILL_MAP = {
    "mean": MeanFill,
    "random": RandomNeighborFill,
    "blur": BlurFill,
    "telea": TeleaFill,
    "patchmatch": PatchMatchFill,
    "resnet": ResNetFill,
    "diffusion": DiffusionFill
}
### 
# maybe also this one:
# from noise_layers.adversarial_attack import AdversarialAttack


def parse_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)


def parse_crop(crop_command):
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))

def parse_cropout(cropout_command):
    matches = re.match(r'cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))


def parse_dropout(dropout_command):
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))

def parse_resize(resize_command):
    matches = re.match(r'resize\((\d+\.*\d*,\d+\.*\d*)\)', resize_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))

def parse_fixed_cnn_inpainting(command):
    """
    Example:
        learnableinpainting(0.1,0.3,32)
    means:
        min_mask_ratio = 0.1
        max_mask_ratio = 0.3
        hidden_channels = 32
    """
    matches = re.match(
        r'learnableinpainting\((\d+\.*\d*),(\d+\.*\d*),(\d+)\)',
        command
    )

    if matches is None:
        raise ValueError(
            f'Invalid learnableinpainting command: {command}'
        )

    min_ratio = float(matches.group(1))
    max_ratio = float(matches.group(2))
    hidden_channels = int(matches.group(3))

    return FixedCNNInpainting(
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        hidden_channels=hidden_channels
    )

# redo mask inpainting, just for evaluation/validation/test
# use telea, randomneighbor, patchmatch,:
def parse_eval_inpainting(inpainting_command: str):
    """
    Example:
    maskinpainting(0.5,10,8,3.0,mean,42)
        max_mask_ratio=0.5,
        max_mask_number=10,
        min_mask_size=8,
        max_aspect_ratio=3.0,
        fill_strategy=None,
        seed=None,
    """

    # 1. basic format check
    if not inpainting_command.startswith("evalinpainting(") or not inpainting_command.endswith(")"):
        raise ValueError(f"Invalid command format: {inpainting_command}")

    # 2. remove wrapper
    content = inpainting_command[len("evalinpainting("):-1]

    # 3. split parameters
    parts = [p.strip() for p in content.split(",")]

    if len(parts) != 6:
        raise ValueError(
            "Expected format: "
            "evalinpainting(max_ratio,max_num,min_size,max_ar,fill,seed)"
        )

    # 4. parse parameters
    max_ratio = float(parts[0])
    max_num = int(parts[1])
    min_size = int(parts[2])
    max_ar = float(parts[3])
    fill_name = parts[4]
    seed = int(parts[5])

    # 5. fill strategy
    if fill_name not in FILL_MAP:
        raise ValueError(f"Unknown fill strategy: {fill_name}")

    fill_strategy = FILL_MAP[fill_name]()

    print("DEBUG fill_strategy:", type(fill_strategy), fill_strategy)
    # 6. build model
    return EvalInpainting(
        max_mask_ratio=max_ratio,
        max_mask_number=max_num,
        min_mask_size=min_size,
        max_aspect_ratio=max_ar,
        fill_strategy=fill_strategy,
        seed=seed,
    )


def parse_haar_wavelet(haarwavelet_command):
    matches = re.match(r'haarwavelet\((\d+\.*\d*)\)', haarwavelet_command)
    strength = float(matches.groups()[0])

    return HaarWavelet(
        strength=strength,
        mode="attenuate",
        attack_bands=("LH", "HL", "HH")
    )

def parse_gaussian_blur(gaussian_blur_command):
    matches = re.match(
        r'gaussianblur\((\d+\.*\d*)(,\d+\.*\d*)?\)',
        gaussian_blur_command
    )

    first = float(matches.groups()[0])
    second = matches.groups()[1]

    if second is None:
        # Fixed sigma, useful for eval / sweep
        return GaussianBlur(
            sigma=first
        )
    else:
        # Random sigma range, useful for training
        second = float(second.replace(',', ''))

        return GaussianBlur(
            sigma=None,
            sigma_range=(first, second)
        )


class NoiseArgParser(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )

    @staticmethod
    def parse_cropout_args(cropout_args):
        pass

    @staticmethod
    def parse_dropout_args(dropout_args):
        pass

    def __call__(self, parser, namespace, values,
                 option_string=None):

        layers = []
        split_commands = values[0].split('+')

        for command in split_commands:
            # remove all whitespace
            command = command.replace(' ', '')
            if command.startswith('cropout'):
                layers.append(parse_cropout(command))
            elif command.startswith('crop'):
                layers.append(parse_crop(command))
            elif command.startswith('dropout'):
                layers.append(parse_dropout(command))
            elif command.startswith('resize'):
                layers.append(parse_resize(command))
            # Keen: adding own method
            elif command.startswith('evalinpainting'):
                layers.append(parse_eval_inpainting(command))
            elif command.startswith('fixedcnninpainting'):
                layers.append(parse_fixed_cnn_inpainting(command))
            elif command.startswith('haarwavelet'):
                layers.append(parse_haar_wavelet(command))
            elif command.startswith('gaussianblur'):
                layers.append(parse_gaussian_blur(command))
            # modified to startwith
            elif command.startswith('jpeg'):
                layers.append('JpegPlaceholder')
            elif command.startswith('quant'):
                layers.append('QuantizationPlaceholder')
            elif command.startswith('identity'):
                # We are adding one Identity() layer in Noiser anyway
                pass
            else:
                raise ValueError('Command not recognized: \n{}'.format(command))
        setattr(namespace, self.dest, layers)
