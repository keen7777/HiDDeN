from __future__ import annotations
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

from mask_optimization import (
    inpaint_hom_diff,
    probabilistic_sparsification,
    nonlocal_pixel_exchange,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Validation images. Images may be located directly inside this
# directory or in subdirectories.
IMAGE_DIR = Path(
    "/home/keen/HiDDeN/images/val"

    
)

# ============================================================
# IMAGE AND MASK CONFIGURATION
# ============================================================

HEIGHT = 128
WIDTH = 128

# Fraction of pixels retained as known pixels.
DENSITY = 0.10

# Homogeneous diffusion parameters.
TAU = 0.25

# Debug values:
DIFFUSION_ITERATIONS = 300
NLPE_ITERATIONS = 200

# Formal experiment values can later be:
# DIFFUSION_ITERATIONS = 150
# NLPE_ITERATIONS = 100

# Probabilistic sparsification parameters.
P = 0.10
Q = 0.10

# NLPE parameters.
N_CANDIDATES = 10

# Random seed used to make the result reproducible.
# fixed to 42
BASE_SEED = 42

# Debug:
#   3    -> process only the first three images
#   None -> process all images
LIMIT = None

# Save partial NumPy results every N processed images.
SAVE_EVERY = 1

OUTPUT_DIR = Path(
    f"/home/keen/HiDDeN/evaluation_data/"
    f"opt_masks_diff_{DIFFUSION_ITERATIONS}"
    f"_nlpe_{NLPE_ITERATIONS}"
    f"_den_{DENSITY}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================

SAVE_VISUALIZATIONS = True

# None: save visualization files for every processed image.
# 10: save only the first 10 visualizations, while all masks
#     are still stored in the NumPy files.
VISUALIZATION_LIMIT = 10

VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

CROP_DIR = VISUALIZATION_DIR / "center_crops"
PS_MASK_DIR = VISUALIZATION_DIR / "ps_masks"
PS_NLPE_MASK_DIR = VISUALIZATION_DIR / "ps_nlpe_masks"

PS_MASKED_DIR = VISUALIZATION_DIR / "ps_masked_images"
PS_NLPE_MASKED_DIR = (
    VISUALIZATION_DIR / "ps_nlpe_masked_images"
)

COMPARISON_DIR = VISUALIZATION_DIR / "comparisons"


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

SUPPORTED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

if SAVE_VISUALIZATIONS:
    for directory in [
        CROP_DIR,
        PS_MASK_DIR,
        PS_NLPE_MASK_DIR,
        PS_MASKED_DIR,
        PS_NLPE_MASKED_DIR,
        COMPARISON_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def collect_image_paths(image_dir: Path) -> list[Path]:
    """
    Recursively collect supported image files in deterministic
    lexicographical order.
    """

    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in: {image_dir}"
        )

    return image_paths


def load_center_crop(
    image_path: Path,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Load one image and apply the same CenterCrop used by the
    validation DataLoader.

    Returns
    -------
    np.ndarray
        RGB array with shape [height, width, 3].
        dtype: float32
        range: [0, 1]
    """

    center_crop = transforms.CenterCrop(
        (height, width)
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = center_crop(image)

        image_np = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

    expected_shape = (
        height,
        width,
        3,
    )

    if image_np.shape != expected_shape:
        raise ValueError(
            f"Unexpected crop shape for {image_path}: "
            f"{image_np.shape}; expected {expected_shape}"
        )

    return image_np


def generate_optimized_masks(
    image: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate:
        1. probabilistic sparsification mask
        2. PS + NLPE optimized mask

    Mask convention:
        True  = known/retained pixel
        False = missing/inpainted pixel
    """

    rng = np.random.default_rng(
        seed
    )

    initial_mask = np.ones(
        (HEIGHT, WIDTH),
        dtype=np.bool_,
    )

    spars_mask = probabilistic_sparsification(
        image=image,
        initial_mask=initial_mask,
        density=DENSITY,
        tau=TAU,
        diff_iterations=DIFFUSION_ITERATIONS,
        p=P,
        q=Q,
        rng=rng,
    )

    nlpe_mask = nonlocal_pixel_exchange(
        image=image,
        mask=spars_mask,
        nlpe_iterations=NLPE_ITERATIONS,
        tau=TAU,
        diff_iterations=DIFFUSION_ITERATIONS,
        n_candidates=N_CANDIDATES,
        rng=rng,
    )

    return (
        spars_mask.astype(
            np.bool_,
            copy=False,
        ),
        nlpe_mask.astype(
            np.bool_,
            copy=False,
        ),
    )



def reconstruct_and_measure_mse(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Reconstruct one image with homogeneous diffusion and calculate
    reconstruction MSE.

    The returned metrics are:

    full_mse:
        Mean squared error over all H x W x C values.

    missing_mse:
        Mean squared error only over pixels where mask is False.
        Since known pixels are copied exactly, this value is usually
        larger than full_mse.

    Returns
    -------
    reconstruction:
        Reconstructed RGB image with shape [H, W, 3].

    full_mse:
        MSE over the complete image.

    missing_mse:
        MSE over the missing/inpainted pixels only.
    """

    reconstruction = inpaint_hom_diff(
        known_image_data=image,
        mask=mask,
        num_iterations=DIFFUSION_ITERATIONS,
        tau=TAU,
    )

    reconstruction = np.asarray(
        reconstruction,
        dtype=np.float64,
    )

    reference = np.asarray(
        image,
        dtype=np.float64,
    )

    if reconstruction.shape != reference.shape:
        raise ValueError(
            "Reconstruction shape does not match image shape: "
            f"reconstruction={reconstruction.shape}, "
            f"image={reference.shape}"
        )

    squared_error = (
        reconstruction - reference
    ) ** 2

    full_mse = float(
        np.mean(squared_error)
    )

    missing_mask = ~mask

    if np.any(missing_mask):
        missing_mse = float(
            np.mean(
                squared_error[missing_mask]
            )
        )
    else:
        missing_mse = 0.0

    return (
        reconstruction,
        full_mse,
        missing_mse,
    )


def float_rgb_to_pil(
    image: np.ndarray,
) -> Image.Image:
    """
    Convert an RGB float image in [0, 1] to a PIL image.
    """

    image_uint8 = np.clip(
        image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        image_uint8
    )


def mask_to_pil(
    mask: np.ndarray,
) -> Image.Image:
    """
    Convert a boolean mask to a grayscale PIL image.

    White = known/retained pixel
    Black = missing/inpainted pixel
    """

    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )

    return Image.fromarray(
        mask_uint8
    )


def apply_mask_for_visualization(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Display missing pixels as black.
    """

    masked_image = image.copy()

    masked_image[~mask] = 0.0

    return masked_image


def add_label(
    image: Image.Image,
    label: str,
    label_height: int = 24,
) -> Image.Image:
    """
    Add a small text label above a visualization tile.
    """

    labeled_image = Image.new(
        "RGB",
        (
            image.width,
            image.height + label_height,
        ),
        color="white",
    )

    labeled_image.paste(
        image.convert("RGB"),
        (0, label_height),
    )

    draw = ImageDraw.Draw(
        labeled_image
    )

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

    return labeled_image


def save_comparison_image(
    crop_image: Image.Image,
    spars_mask_image: Image.Image,
    spars_masked_image: Image.Image,
    nlpe_mask_image: Image.Image,
    nlpe_masked_image: Image.Image,
    output_path: Path,
):
    """
    Save a horizontal overview:

    crop | PS mask | PS known pixels | NLPE mask | NLPE known pixels
    """

    tiles = [
        add_label(
            crop_image,
            "Center crop",
        ),
        add_label(
            spars_mask_image,
            "PS mask",
        ),
        add_label(
            spars_masked_image,
            "PS known pixels",
        ),
        add_label(
            nlpe_mask_image,
            "PS + NLPE mask",
        ),
        add_label(
            nlpe_masked_image,
            "PS + NLPE known",
        ),
    ]

    panel_width = sum(
        tile.width for tile in tiles
    )

    panel_height = max(
        tile.height for tile in tiles
    )

    panel = Image.new(
        "RGB",
        (
            panel_width,
            panel_height,
        ),
        color="white",
    )

    current_x = 0

    for tile in tiles:
        panel.paste(
            tile,
            (current_x, 0),
        )

        current_x += tile.width

    panel.save(
        output_path
    )


def save_mask_visualizations(
    image: np.ndarray,
    spars_mask: np.ndarray,
    nlpe_mask: np.ndarray,
    image_path: Path,
    index: int,
):
    """
    Save human-readable PNG visualizations.
    """

    safe_stem = (
        f"{index:04d}_{image_path.stem}"
    )

    crop_image = float_rgb_to_pil(
        image
    )

    spars_mask_image = mask_to_pil(
        spars_mask
    )

    nlpe_mask_image = mask_to_pil(
        nlpe_mask
    )

    spars_masked = (
        apply_mask_for_visualization(
            image,
            spars_mask,
        )
    )

    nlpe_masked = (
        apply_mask_for_visualization(
            image,
            nlpe_mask,
        )
    )

    spars_masked_image = (
        float_rgb_to_pil(
            spars_masked
        )
    )

    nlpe_masked_image = (
        float_rgb_to_pil(
            nlpe_masked
        )
    )

    crop_image.save(
        CROP_DIR
        / f"{safe_stem}_crop.png"
    )

    spars_mask_image.save(
        PS_MASK_DIR
        / f"{safe_stem}_ps_mask.png"
    )

    nlpe_mask_image.save(
        PS_NLPE_MASK_DIR
        / f"{safe_stem}_ps_nlpe_mask.png"
    )

    spars_masked_image.save(
        PS_MASKED_DIR
        / f"{safe_stem}_ps_masked.png"
    )

    nlpe_masked_image.save(
        PS_NLPE_MASKED_DIR
        / f"{safe_stem}_ps_nlpe_masked.png"
    )

    save_comparison_image(
        crop_image=crop_image,
        spars_mask_image=spars_mask_image,
        spars_masked_image=spars_masked_image,
        nlpe_mask_image=nlpe_mask_image,
        nlpe_masked_image=nlpe_masked_image,
        output_path=(
            COMPARISON_DIR
            / f"{safe_stem}_comparison.png"
        ),
    )


def save_progress(
    nlpe_masks: np.ndarray,
    spars_masks: np.ndarray,
    image_ids: list[str],
    mask_mapping: list[dict],
    mse_records: list[dict],
    processed_count: int,
):
    """
    Save intermediate results so that already computed masks are
    not completely lost if the script is interrupted.
    """

    np.save(
        OUTPUT_DIR
        / "partial_val_ps_nlpe_masks.npy",
        nlpe_masks[:processed_count],
    )

    np.save(
        OUTPUT_DIR
        / "partial_val_ps_masks.npy",
        spars_masks[:processed_count],
    )

    with open(
        OUTPUT_DIR
        / "partial_val_image_ids.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            image_ids,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        OUTPUT_DIR
        / "partial_val_mask_mapping.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            mask_mapping,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        OUTPUT_DIR
        / "partial_val_reconstruction_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            mse_records,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    image_paths = collect_image_paths(
        IMAGE_DIR
    )

    total_found = len(
        image_paths
    )

    print(
        f"Found {total_found} images before limit"
    )

    if LIMIT is not None:
        if LIMIT <= 0:
            raise ValueError(
                "LIMIT must be a positive integer or None."
            )

        image_paths = image_paths[:LIMIT]

    num_images = len(
        image_paths
    )

    print(
        f"Processing {num_images} images"
    )
    print(
        f"Crop size: {HEIGHT}x{WIDTH}"
    )
    print(
        f"Target density: {DENSITY:.4f}"
    )
    print(
        f"Diffusion iterations: "
        f"{DIFFUSION_ITERATIONS}"
    )
    print(
        f"NLPE iterations: "
        f"{NLPE_ITERATIONS}"
    )
    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )
    print(
        f"Save visualizations: "
        f"{SAVE_VISUALIZATIONS}"
    )

    if (
        LIMIT is None
        and total_found != 300
    ):
        print(
            "Warning: expected 300 validation images, "
            f"but found {total_found}"
        )

    spars_masks = np.empty(
        (
            num_images,
            HEIGHT,
            WIDTH,
        ),
        dtype=np.bool_,
    )

    nlpe_masks = np.empty(
        (
            num_images,
            HEIGHT,
            WIDTH,
        ),
        dtype=np.bool_,
    )

    image_ids = []
    mask_mapping = []
    elapsed_times = []
    mse_records = []

    ps_mse_values = []
    nlpe_mse_values = []
    ps_missing_mse_values = []
    nlpe_missing_mse_values = []
    relative_improvement_values = []

    total_start = time.perf_counter()

    for index, image_path in enumerate(
        tqdm(
            image_paths,
            desc="Images",
            unit="image",
        )
    ):
        image_start = time.perf_counter()

        image_np = load_center_crop(
            image_path=image_path,
            height=HEIGHT,
            width=WIDTH,
        )

        spars_mask, nlpe_mask = (
            generate_optimized_masks(
                image=image_np,
                seed=BASE_SEED + index,
            )
        )

        spars_masks[index] = spars_mask
        nlpe_masks[index] = nlpe_mask

        relative_path = str(
            image_path.relative_to(
                IMAGE_DIR
            )
        )

        image_ids.append(
            relative_path
        )

        # Explicitly record the correspondence between each row in
        # the saved mask arrays and the source validation image.
        # The actual per-image seed is also stored because generation
        # uses BASE_SEED + index rather than the same seed for all images.
        mask_mapping.append(
            {
                "mask_index": index,
                "image_id": relative_path,
                "image_name": image_path.name,
                "seed": BASE_SEED + index,
            }
        )

        # Reconstruct the same image with the PS and PS+NLPE masks
        # using the same homogeneous diffusion settings. This gives
        # a fair post-hoc comparison of their reconstruction quality.
        (
            _,
            ps_mse,
            ps_missing_mse,
        ) = reconstruct_and_measure_mse(
            image=image_np,
            mask=spars_mask,
        )

        (
            _,
            nlpe_mse,
            nlpe_missing_mse,
        ) = reconstruct_and_measure_mse(
            image=image_np,
            mask=nlpe_mask,
        )

        absolute_improvement = (
            ps_mse - nlpe_mse
        )

        if ps_mse > 0.0:
            relative_improvement_percent = (
                absolute_improvement
                / ps_mse
                * 100.0
            )
        else:
            relative_improvement_percent = 0.0

        ps_mse_values.append(
            ps_mse
        )
        nlpe_mse_values.append(
            nlpe_mse
        )
        ps_missing_mse_values.append(
            ps_missing_mse
        )
        nlpe_missing_mse_values.append(
            nlpe_missing_mse
        )
        relative_improvement_values.append(
            relative_improvement_percent
        )

        mse_records.append(
            {
                "index": index,
                "image_id": relative_path,
                "ps_full_mse": ps_mse,
                "ps_nlpe_full_mse": nlpe_mse,
                "absolute_full_mse_improvement": (
                    absolute_improvement
                ),
                "relative_full_mse_improvement_percent": (
                    relative_improvement_percent
                ),
                "ps_missing_region_mse": (
                    ps_missing_mse
                ),
                "ps_nlpe_missing_region_mse": (
                    nlpe_missing_mse
                ),
            }
        )

        if nlpe_mse > ps_mse + 1e-12:
            print(
                "\nWarning: PS+NLPE reconstruction MSE "
                "is larger than PS reconstruction MSE for "
                f"{relative_path}"
            )

        if SAVE_VISUALIZATIONS:
            should_save_visualization = (
                VISUALIZATION_LIMIT is None
                or index < VISUALIZATION_LIMIT
            )

            if should_save_visualization:
                save_mask_visualizations(
                    image=image_np,
                    spars_mask=spars_mask,
                    nlpe_mask=nlpe_mask,
                    image_path=image_path,
                    index=index,
                )

        elapsed = (
            time.perf_counter()
            - image_start
        )

        elapsed_times.append(
            elapsed
        )

        spars_density = float(
            spars_mask.mean()
        )

        nlpe_density = float(
            nlpe_mask.mean()
        )

        print(
            f"\n[{index + 1}/{num_images}] "
            f"{relative_path}"
        )
        print(
            f"PS density: "
            f"{spars_density:.6f}"
        )
        print(
            f"PS+NLPE density: "
            f"{nlpe_density:.6f}"
        )
        print(
            f"PS reconstruction MSE: "
            f"{ps_mse:.10f}"
        )
        print(
            f"PS+NLPE reconstruction MSE: "
            f"{nlpe_mse:.10f}"
        )
        print(
            f"Absolute MSE improvement: "
            f"{absolute_improvement:.10f}"
        )
        print(
            f"Relative MSE improvement: "
            f"{relative_improvement_percent:.4f}%"
        )
        print(
            f"PS missing-region MSE: "
            f"{ps_missing_mse:.10f}"
        )
        print(
            f"PS+NLPE missing-region MSE: "
            f"{nlpe_missing_mse:.10f}"
        )
        print(
            f"Time: {elapsed:.2f} seconds"
        )

        target_known_pixels = int(
            DENSITY * HEIGHT * WIDTH
        )

        actual_known_pixels = int(
            nlpe_mask.sum()
        )

        if (
            actual_known_pixels
            != target_known_pixels
        ):
            print(
                "Warning: known-pixel count differs "
                "from the target: "
                f"target={target_known_pixels}, "
                f"actual={actual_known_pixels}"
            )

        processed_count = index + 1

        if (
            processed_count % SAVE_EVERY == 0
            or processed_count == num_images
        ):
            save_progress(
                nlpe_masks=nlpe_masks,
                spars_masks=spars_masks,
                image_ids=image_ids,
                mask_mapping=mask_mapping,
                mse_records=mse_records,
                processed_count=processed_count,
            )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    # 0.10 -> 100
    density_code = int(
        round(DENSITY * 1000)
    )

    ps_mask_file = OUTPUT_DIR / (
        f"val_ps_masks_density_"
        f"{density_code:03d}.npy"
    )

    nlpe_mask_file = OUTPUT_DIR / (
        f"val_ps_nlpe_masks_density_"
        f"{density_code:03d}.npy"
    )

    ids_file = OUTPUT_DIR / (
        f"val_image_ids_density_"
        f"{density_code:03d}.json"
    )

    mapping_file = OUTPUT_DIR / (
        f"val_mask_mapping_density_"
        f"{density_code:03d}.json"
    )

    config_file = OUTPUT_DIR / (
        f"val_config_density_"
        f"{density_code:03d}.json"
    )

    metrics_file = OUTPUT_DIR / (
        f"val_reconstruction_metrics_density_"
        f"{density_code:03d}.json"
    )

    np.save(
        ps_mask_file,
        spars_masks,
    )

    np.save(
        nlpe_mask_file,
        nlpe_masks,
    )

    with open(
        ids_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            image_ids,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        mapping_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            mask_mapping,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            mse_records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    config = {
        "image_dir": str(
            IMAGE_DIR
        ),
        "output_dir": str(
            OUTPUT_DIR
        ),
        "total_images_found": (
            total_found
        ),
        "processed_images": (
            num_images
        ),
        "limit": LIMIT,
        "height": HEIGHT,
        "width": WIDTH,
        "crop_method": (
            "torchvision.transforms.CenterCrop"
        ),
        "image_value_range": [
            0.0,
            1.0,
        ],
        "mask_convention": {
            "true": (
                "known/retained pixel"
            ),
            "false": (
                "missing/inpainted pixel"
            ),
        },
        "visualization_convention": {
            "white": (
                "known/retained pixel"
            ),
            "black": (
                "missing/inpainted pixel"
            ),
        },
        "density": DENSITY,
        "target_known_pixels": int(
            DENSITY * HEIGHT * WIDTH
        ),
        "tau": TAU,
        "diffusion_iterations": (
            DIFFUSION_ITERATIONS
        ),
        "p": P,
        "q": Q,
        "nlpe_iterations": (
            NLPE_ITERATIONS
        ),
        "n_candidates": (
            N_CANDIDATES
        ),
        "base_seed": BASE_SEED,
        "save_visualizations": (
            SAVE_VISUALIZATIONS
        ),
        "visualization_limit": (
            VISUALIZATION_LIMIT
        ),
        "mean_seconds_per_image": (
            float(
                np.mean(elapsed_times)
            )
            if elapsed_times
            else None
        ),
        "median_seconds_per_image": (
            float(
                np.median(elapsed_times)
            )
            if elapsed_times
            else None
        ),
        "total_seconds": float(
            total_elapsed
        ),
        "mean_ps_density": float(
            spars_masks.mean()
        ),
        "mean_ps_nlpe_density": float(
            nlpe_masks.mean()
        ),
        "mean_ps_full_mse": (
            float(np.mean(ps_mse_values))
            if ps_mse_values
            else None
        ),
        "mean_ps_nlpe_full_mse": (
            float(np.mean(nlpe_mse_values))
            if nlpe_mse_values
            else None
        ),
        "mean_ps_missing_region_mse": (
            float(np.mean(ps_missing_mse_values))
            if ps_missing_mse_values
            else None
        ),
        "mean_ps_nlpe_missing_region_mse": (
            float(np.mean(nlpe_missing_mse_values))
            if nlpe_missing_mse_values
            else None
        ),
        "mean_relative_full_mse_improvement_percent": (
            float(np.mean(relative_improvement_values))
            if relative_improvement_values
            else None
        ),
        "median_relative_full_mse_improvement_percent": (
            float(np.median(relative_improvement_values))
            if relative_improvement_values
            else None
        ),
        "images_with_non_worse_nlpe_mse": int(
            sum(
                nlpe <= ps + 1e-12
                for ps, nlpe in zip(
                    ps_mse_values,
                    nlpe_mse_values,
                )
            )
        ),
    }

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Delete partial files only after the full run completed.
    temporary_files = [
        OUTPUT_DIR
        / "partial_val_ps_nlpe_masks.npy",
        OUTPUT_DIR
        / "partial_val_ps_masks.npy",
        OUTPUT_DIR
        / "partial_val_image_ids.json",
        OUTPUT_DIR
        / "partial_val_reconstruction_metrics.json",
        OUTPUT_DIR
        / "partial_val_mask_mapping.json",
    ]

    for temporary_file in temporary_files:
        if temporary_file.exists():
            temporary_file.unlink()

    print("\n====================================")
    print("PRECOMPUTATION COMPLETED")
    print("------------------------------------")
    print(
        f"Processed images: "
        f"{num_images}"
    )
    print(
        f"PS mask shape: "
        f"{spars_masks.shape}"
    )
    print(
        f"PS+NLPE mask shape: "
        f"{nlpe_masks.shape}"
    )
    print(
        f"Mask dtype: "
        f"{nlpe_masks.dtype}"
    )
    print(
        f"Mean PS density: "
        f"{spars_masks.mean():.6f}"
    )
    print(
        f"Mean PS+NLPE density: "
        f"{nlpe_masks.mean():.6f}"
    )

    if ps_mse_values:
        mean_ps_mse = float(
            np.mean(ps_mse_values)
        )
        mean_nlpe_mse = float(
            np.mean(nlpe_mse_values)
        )
        mean_relative_improvement = float(
            np.mean(relative_improvement_values)
        )

        print(
            f"Mean PS reconstruction MSE: "
            f"{mean_ps_mse:.10f}"
        )
        print(
            f"Mean PS+NLPE reconstruction MSE: "
            f"{mean_nlpe_mse:.10f}"
        )
        print(
            f"Mean relative MSE improvement: "
            f"{mean_relative_improvement:.4f}%"
        )
        print(
            "Images where PS+NLPE MSE is not worse: "
            f"{sum(nlpe <= ps + 1e-12 for ps, nlpe in zip(ps_mse_values, nlpe_mse_values))}"
            f"/{len(ps_mse_values)}"
        )

    if elapsed_times:
        print(
            f"Mean time per image: "
            f"{np.mean(elapsed_times):.2f} seconds"
        )
        print(
            f"Median time per image: "
            f"{np.median(elapsed_times):.2f} seconds"
        )

    print(
        f"Total time: "
        f"{total_elapsed / 60:.2f} minutes"
    )
    print("------------------------------------")
    print(
        f"PS masks: "
        f"{ps_mask_file}"
    )
    print(
        f"PS+NLPE masks: "
        f"{nlpe_mask_file}"
    )
    print(
        f"Image IDs: "
        f"{ids_file}"
    )
    print(
        f"Mask mapping: "
        f"{mapping_file}"
    )
    print(
        f"Config: "
        f"{config_file}"
    )
    print(
        f"Reconstruction metrics: "
        f"{metrics_file}"
    )

    if SAVE_VISUALIZATIONS:
        print(
            f"Visualizations: "
            f"{VISUALIZATION_DIR}"
        )
        print(
            f"Comparison panels: "
            f"{COMPARISON_DIR}"
        )

    print("====================================")


if __name__ == "__main__":
    main()