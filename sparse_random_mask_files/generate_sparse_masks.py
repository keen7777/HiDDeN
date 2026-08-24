#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# CONFIGURATION
# ============================================================

# Input images
IMAGE_DIR = Path("/home/keen/HiDDeN/images/val/val_class")

# Output root
OUTPUT_ROOT = Path(
    "/home/keen/HiDDeN/sparse_random_masks_data"
)

# Image size (must match your evaluation setup)
IMAGE_SIZE = 128

# Densities to generate
# 0.1 -> 10% retained pixels
# 0.3 -> 30% retained pixels
DENSITIES = [0.1, 0.3]

# Seed for reproducibility
SEED = 42

# Supported image suffixes
SUPPORTED_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}

# Visualization
SAVE_VISUALIZATIONS = True

# Save only the first N examples
# None means save all
VISUALIZATION_LIMIT = 10


# ============================================================
# IMAGE HELPERS
# ============================================================

def collect_image_paths(image_dir: Path):
    """
    Collect and sort image files from a directory.
    """
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}"
        )

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No supported image files found in: {image_dir}"
        )

    return image_paths


def load_center_crop_rgb(
    image_path: Path,
    image_size: int = 128,
) -> np.ndarray:
    """
    Load image as RGB and center-crop/resize to image_size x image_size.

    Returns:
        np.ndarray of shape [H, W, 3], float32 in [0, 1]
    """
    image = Image.open(image_path).convert("RGB")

    # ImageOps.fit does center crop + resize
    image = ImageOps.fit(
        image,
        (image_size, image_size),
        method=Image.Resampling.LANCZOS,
    )

    image_np = np.asarray(image).astype(np.float32) / 255.0
    return image_np


# ============================================================
# RANDOM SPARSE MASK GENERATION
# ============================================================

def generate_random_sparse_mask(
    height: int,
    width: int,
    retained_density: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Generate one random sparse mask.

    Stored mask convention:
        1 = retained / known
        0 = removed / reconstructed
    """
    total_pixels = height * width
    retained_count = int(
        round(retained_density * total_pixels)
    )

    retained_count = max(
        1,
        min(retained_count, total_pixels),
    )

    flat_mask = np.zeros(
        total_pixels,
        dtype=np.uint8,
    )

    retained_indices = rng.choice(
        total_pixels,
        size=retained_count,
        replace=False,
    )

    flat_mask[retained_indices] = 1

    mask = flat_mask.reshape(height, width)
    return mask


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def float_rgb_to_pil(
    image: np.ndarray,
) -> Image.Image:
    """
    Convert an RGB float image in [0, 1] to PIL.
    """
    image_uint8 = np.clip(
        image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(image_uint8)


def mask_to_pil(
    mask: np.ndarray,
) -> Image.Image:
    """
    Convert mask to grayscale PIL.

    White = known / retained
    Black = missing / reconstructed
    """
    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )
    return Image.fromarray(mask_uint8, mode="L")


def apply_mask_for_visualization(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Visualize known pixels only.

    Mask convention:
        1 = retained / known
        0 = removed / reconstructed

    Missing pixels are shown as black.
    """
    masked_image = image.copy()
    masked_image[mask == 0] = 0.0
    return masked_image


def add_label(
    image: Image.Image,
    label: str,
    label_height: int = 24,
) -> Image.Image:
    """
    Add a small label above an image tile.
    """
    labeled = Image.new(
        "RGB",
        (
            image.width,
            image.height + label_height,
        ),
        color="white",
    )

    labeled.paste(
        image.convert("RGB"),
        (0, label_height),
    )

    draw = ImageDraw.Draw(labeled)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text(
        (4, 5),
        label,
        fill="black",
        font=font,
    )

    return labeled


def save_comparison_image(
    crop_image: Image.Image,
    random_mask_image: Image.Image,
    random_masked_image: Image.Image,
    output_path: Path,
):
    """
    Save side-by-side comparison:
        crop | mask | masked image
    """
    tiles = [
        add_label(crop_image, "Center crop"),
        add_label(random_mask_image, "Random sparse mask"),
        add_label(random_masked_image, "Random known pixels"),
    ]

    panel_width = sum(
        tile.width for tile in tiles
    )
    panel_height = max(
        tile.height for tile in tiles
    )

    panel = Image.new(
        "RGB",
        (panel_width, panel_height),
        color="white",
    )

    current_x = 0
    for tile in tiles:
        panel.paste(tile, (current_x, 0))
        current_x += tile.width

    panel.save(output_path)


def save_random_mask_visualizations(
    image: np.ndarray,
    random_mask: np.ndarray,
    image_path: Path,
    index: int,
    crop_dir: Path,
    random_mask_dir: Path,
    random_masked_dir: Path,
    comparison_dir: Path,
):
    """
    Save visualization files for one sample.
    """
    safe_stem = (
        f"{index:04d}_{image_path.stem}"
    )

    crop_image = float_rgb_to_pil(image)
    random_mask_image = mask_to_pil(random_mask)

    random_masked = apply_mask_for_visualization(
        image=image,
        mask=random_mask,
    )
    random_masked_image = float_rgb_to_pil(
        random_masked
    )

    crop_image.save(
        crop_dir / f"{safe_stem}_crop.png"
    )

    random_mask_image.save(
        random_mask_dir
        / f"{safe_stem}_random_mask.png"
    )

    random_masked_image.save(
        random_masked_dir
        / f"{safe_stem}_random_masked.png"
    )

    save_comparison_image(
        crop_image=crop_image,
        random_mask_image=random_mask_image,
        random_masked_image=random_masked_image,
        output_path=(
            comparison_dir
            / f"{safe_stem}_comparison.png"
        ),
    )


# ============================================================
# JSON HELPERS
# ============================================================

def save_json(
    obj,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# MAIN GENERATION
# ============================================================

def main():
    print("=" * 60)
    print("RANDOM SPARSE MASK PRECOMPUTATION")
    print("=" * 60)
    print(f"Image directory : {IMAGE_DIR}")
    print(f"Output root     : {OUTPUT_ROOT}")
    print(f"Image size      : {IMAGE_SIZE}")
    print(f"Densities       : {DENSITIES}")
    print(f"Seed            : {SEED}")
    print(f"Visualizations  : {SAVE_VISUALIZATIONS}")
    print(
        f"Visualization limit: {VISUALIZATION_LIMIT}"
    )
    print("=" * 60)

    image_paths = collect_image_paths(IMAGE_DIR)
    print(
        f"Found {len(image_paths)} input images."
    )

    # fixed ordering
    image_ids = [
        path.stem for path in image_paths
    ]

    for density in DENSITIES:
        density_tag = f"{density:.1f}"
        density_int = int(round(density * 1000))

        run_dir = (
            OUTPUT_ROOT
            / f"random_masks_den_{density_tag}_seed_{SEED}"
        )
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        masks = []
        mapping = []

        # Visualization directories
        if SAVE_VISUALIZATIONS:
            visualization_dir = (
                run_dir / "visualizations"
            )
            crop_dir = (
                visualization_dir / "center_crops"
            )
            random_mask_dir = (
                visualization_dir / "random_masks"
            )
            random_masked_dir = (
                visualization_dir
                / "random_masked_images"
            )
            comparison_dir = (
                visualization_dir / "comparisons"
            )

            for directory in [
                crop_dir,
                random_mask_dir,
                random_masked_dir,
                comparison_dir,
            ]:
                directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

        print("-" * 60)
        print(
            f"Generating masks for retained density = {density:.3f}"
        )
        print("-" * 60)

        # use a density-specific RNG so each density is deterministic
        rng = np.random.RandomState(
            SEED + density_int
        )

        for index, image_path in enumerate(image_paths):
            image_np = load_center_crop_rgb(
                image_path=image_path,
                image_size=IMAGE_SIZE,
            )

            mask = generate_random_sparse_mask(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                retained_density=density,
                rng=rng,
            )

            masks.append(mask)

            retained_density_actual = float(
                mask.mean()
            )
            removal_ratio_actual = float(
                1.0 - retained_density_actual
            )

            mapping.append(
                {
                    "index": index,
                    "image_id": image_path.stem,
                    "image_filename": image_path.name,
                    "mask_filename": (
                        f"mask_{index:04d}.npy"
                    ),
                    "retained_density": (
                        retained_density_actual
                    ),
                    "removal_ratio": (
                        removal_ratio_actual
                    ),
                }
            )

            if SAVE_VISUALIZATIONS:
                should_save_visualization = (
                    VISUALIZATION_LIMIT is None
                    or index < VISUALIZATION_LIMIT
                )

                if should_save_visualization:
                    save_random_mask_visualizations(
                        image=image_np,
                        random_mask=mask,
                        image_path=image_path,
                        index=index,
                        crop_dir=crop_dir,
                        random_mask_dir=random_mask_dir,
                        random_masked_dir=random_masked_dir,
                        comparison_dir=comparison_dir,
                    )

            if (index + 1) % 50 == 0:
                print(
                    f"Processed {index + 1}/{len(image_paths)}"
                )

        masks = np.stack(
            masks,
            axis=0,
        ).astype(np.uint8)

        mean_retained_density = float(
            masks.mean()
        )
        mean_removal_ratio = float(
            1.0 - mean_retained_density
        )

        mask_file = (
            run_dir
            / f"val_random_masks_density_{density_int}.npy"
        )
        image_ids_file = (
            run_dir
            / f"val_image_ids_density_{density_int}.json"
        )
        mapping_file = (
            run_dir
            / f"val_mask_mapping_density_{density_int}.json"
        )
        config_file = (
            run_dir
            / f"val_config_density_{density_int}.json"
        )

        np.save(mask_file, masks)

        save_json(
            image_ids,
            image_ids_file,
        )

        save_json(
            mapping,
            mapping_file,
        )

        config = {
            "mask_type": "random_sparse",
            "mask_count": int(masks.shape[0]),
            "mask_shape": list(masks.shape),
            "image_size": IMAGE_SIZE,
            "seed": SEED,
            "target_retained_density": density,
            "target_removal_ratio": 1.0 - density,
            "mean_retained_density": mean_retained_density,
            "mean_removal_ratio": mean_removal_ratio,
            "stored_mask_convention": (
                "1 = retained/known, 0 = removed/reconstructed"
            ),
            "image_directory": str(IMAGE_DIR),
            "output_directory": str(run_dir),
            "visualizations_saved": bool(
                SAVE_VISUALIZATIONS
            ),
            "visualization_limit": VISUALIZATION_LIMIT,
        }

        save_json(
            config,
            config_file,
        )

        print()
        print(
            f"Finished density {density:.3f}"
        )
        print(
            f"Saved masks      : {mask_file}"
        )
        print(
            f"Saved image ids  : {image_ids_file}"
        )
        print(
            f"Saved mapping    : {mapping_file}"
        )
        print(
            f"Saved config     : {config_file}"
        )
        print(
            f"Mean retained density : {mean_retained_density:.6f}"
        )
        print(
            f"Mean removal ratio    : {mean_removal_ratio:.6f}"
        )

        if SAVE_VISUALIZATIONS:
            print(
                f"Visualizations root   : {visualization_dir}"
            )
            print(
                f"Comparison panels     : {comparison_dir}"
            )

        print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()