import json
from pathlib import Path

import pandas as pd


# ============================================================
# paths
# ============================================================

PROJECT_ROOT = Path("/home/keen/HiDDeN")
OUTPUT = PROJECT_ROOT / "output"

CSV_DIR = OUTPUT / "csv"
TEX_DIR = OUTPUT / "tex"

CSV_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OPT_ROOT = (
    PROJECT_ROOT
    / "optimized_mask_eval_outputs_metrics"
)

RANDOM_ROOT = (
    PROJECT_ROOT
    / "sparse_random_mask_eval_outputs_metrics"
)


# ============================================================
# configs
# ============================================================

MODELS = {
    "Baseline":
        "3k_baseline",

    "Gaussian Blur":
        "3k_gaussian_blur",

    "Haar Wavelet":
        "3k_haar_wavelet",

    "Frozen UNet":
        "3k_frozen_pretrained_unet_eval_aligned",
}


MODEL_MAP = {
    raw_name: display_name
    for display_name, raw_name
    in MODELS.items()
}


ATTACKS = [
    "mean",
    "telea",
    "navier",
    "diffusion",
]


DENSITIES = [
    0.1,
    0.3,
]


MASK_TYPES = [
    "Random",
    "O150",
    "O300",
]


EXPECTED_ROWS = (
    len(MODELS)
    * len(ATTACKS)
    * len(DENSITIES)
    * len(MASK_TYPES)
)


# ============================================================
# helpers
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def find_all_result_files():
    files = []

    for path in OPT_ROOT.rglob(
        "evaluation_results.json"
    ):
        if "opt_masks_diff_" in str(
            path
        ):
            files.append(path)

    for path in RANDOM_ROOT.rglob(
        "evaluation_results.json"
    ):
        if "random_masks_den_" in str(
            path
        ):
            files.append(path)

    return sorted(files)


def require_metric(data, key, source):
    if key not in data:
        raise KeyError(
            f"Missing '{key}' in {source}. "
            "This collector expects the new metrics-format "
            "evaluation output."
        )

    value = data[key]

    if value is None:
        raise ValueError(
            f"Metric '{key}' is null in {source}."
        )

    return value


def infer_mask_type(folder):
    if (
        "opt_masks_diff_150_nlpe_100"
        in folder
    ):
        return "O150"

    if (
        "opt_masks_diff_300_nlpe_200"
        in folder
    ):
        return "O300"

    if "random_masks_den_" in folder:
        return "Random"

    return None


def infer_density(folder):
    if "density_100" in folder:
        return 0.1

    if "density_300" in folder:
        return 0.3

    return None


# ============================================================
# collect
# ============================================================

def collect_sparse_results():
    """
    Collect both image-quality definitions:

        final:
            cover -> attacked
            PSNR_final / SSIM_final

        attack-only:
            encoded -> attacked
            PSNR_attack / SSIM_attack

    O150 and O300 refer to the mask-generation configuration,
    not to the final evaluation-time diffusion iteration count.
    """

    rows = []

    files = find_all_result_files()

    print(
        "Found files:",
        len(files),
    )

    for path in files:

        data = load_json(path)

        folder = str(
            path.parent
        )

        # -----------------------
        # model
        # -----------------------

        model_raw = data["model_name"]

        model = MODEL_MAP.get(
            model_raw,
            model_raw,
        )

        # -----------------------
        # attack
        # -----------------------

        attack = data["attack"]

        if attack not in ATTACKS:
            continue

        # -----------------------
        # mask type
        # -----------------------

        mask_type = infer_mask_type(
            folder
        )

        if mask_type is None:
            continue

        # -----------------------
        # density
        # -----------------------

        density = infer_density(
            folder
        )

        if density is None:
            print(
                "unknown density:",
                folder,
            )
            continue

        rows.append(
            {
                "model":
                    model,

                "mask_type":
                    mask_type,

                "density":
                    density,

                "attack":
                    attack,

                "PSNR_final":
                    require_metric(
                        data,
                        "psnr_final",
                        path,
                    ),

                "SSIM_final":
                    require_metric(
                        data,
                        "ssim_final",
                        path,
                    ),

                "PSNR_attack":
                    require_metric(
                        data,
                        "psnr_attack",
                        path,
                    ),

                "SSIM_attack":
                    require_metric(
                        data,
                        "ssim_attack",
                        path,
                    ),

                "BER":
                    require_metric(
                        data,
                        "ber",
                        path,
                    ),

                "num_images":
                    data["processed_images"],
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            [
                "model",
                "attack",
                "density",
                "mask_type",
            ]
        ).reset_index(drop=True)

        duplicate_columns = [
            "model",
            "attack",
            "density",
            "mask_type",
        ]

        duplicates = df.duplicated(
            subset=duplicate_columns,
            keep=False,
        )

        if duplicates.any():
            raise RuntimeError(
                "Duplicate sparse-result conditions found:\n"
                + df.loc[
                    duplicates,
                    duplicate_columns,
                ].to_string(
                    index=False
                )
            )

    return df


# ============================================================
# latex
# ============================================================

def save_latex(df):

    latex = df.to_latex(
        index=False,
        escape=True,
    )

    content = f"""
\\begin{{table}}[h]
\\centering

\\caption{{Sparse-mask robustness evaluation. Final image quality is measured against the cover image; attack-only quality is measured against the encoded image.}}

\\label{{tab:sparse_mask_results}}

{latex}

\\end{{table}}
"""

    with open(
        TEX_DIR
        / "sparse_mask_results.tex",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(content)


# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    print(
        "Collect sparse mask results..."
    )

    df = collect_sparse_results()

    print()
    print(
        "Collected rows:",
        len(df),
    )

    if len(df) != EXPECTED_ROWS:
        print(
            "WARNING: expected "
            f"{EXPECTED_ROWS} rows!"
        )
    else:
        print(
            "Sanity check passed."
        )

    print()
    print(
        df.groupby(
            [
                "mask_type",
                "density",
            ]
        ).size()
    )

    df.to_csv(
        CSV_DIR
        / "sparse_mask_results.csv",
        index=False,
    )

    save_latex(df)

    print()
    print("Saved:")

    print(
        CSV_DIR
        / "sparse_mask_results.csv"
    )

    print(
        TEX_DIR
        / "sparse_mask_results.tex"
    )
