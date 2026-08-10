from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class OptimizedMaskTestDataset(Dataset):
    """
    Validation dataset for optimized-mask evaluation.

    Each image is center-cropped deterministically and paired with
    one precomputed optimized mask.

    Returns:
        image: Tensor [3, H, W], normalized to [-1, 1]
        label: int
        mask: Tensor [1, H, W], where 1 means known pixel
        image_id: relative image path
    """

    def __init__(
        self,
        image_folder: str,
        mask_file: str,
        height: int,
        width: int,
    ):
        # ImageFolder discovers images and preserves a deterministic order.
        self.base_dataset = datasets.ImageFolder(
            image_folder,
            transform=None,
        )

        self.image_folder = Path(image_folder)

        self.image_transform = transforms.Compose([
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ])

        self.masks = np.load(
            mask_file,
            mmap_mode="r",
        )

        if len(self.base_dataset) != len(self.masks):
            raise ValueError(
                "The number of validation images and masks differs: "
                f"{len(self.base_dataset)} images versus "
                f"{len(self.masks)} masks."
            )

        expected_shape = (height, width)

        if tuple(self.masks.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected masks with shape "
                f"(N, {height}, {width}), "
                f"but got {self.masks.shape}."
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image_path, label = self.base_dataset.samples[index]

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        mask_np = np.array(
            self.masks[index],
            dtype=np.float32,
            copy=True,
        )

        # [H, W] -> [1, H, W]
        mask = torch.from_numpy(mask_np).unsqueeze(0)

        image_id = str(
            Path(image_path).relative_to(self.image_folder)
        )

        return image, label, mask, image_id