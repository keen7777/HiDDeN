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

CSV_DIR.mkdir(parents=True, exist_ok=True)
TEX_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# model information
# ============================================================

MODELS = {
    "Baseline": {
        "validation":
            PROJECT_ROOT /
            "validate_trained_models_result"/ 
            "evaluation_3k_baseline_results.json",
        "prefix":
            "3k_baseline",
    },

    "Gaussian Blur": {
        "validation":
            PROJECT_ROOT /
            "validate_trained_models_result"/ 
            "evaluation_3k_gaussian_blur_results.json",
        "prefix":
            "3k_gaussian_blur",
    },

    "Haar Wavelet": {
        "validation":
            PROJECT_ROOT /
            "validate_trained_models_result"/
            "evaluation_3k_haar_wavelet_results.json",
        "prefix":
            "3k_haar_wavelet",
    },

    "Frozen UNet": {
        "validation":
            PROJECT_ROOT /
            "validate_trained_models_result"/ "evaluation_3k_frozen_pretrained_unet_eval_aligned_results.json",
        "prefix":
            "3k_frozen_pretrained_unet_eval_aligned",
    },
}


# ============================================================
# rectangular sweep
# ============================================================

ATTACKS = [
    "mean",
    "telea",
    "navier",
    "shiftmap",
    "diffusion",
]

SWEEP_ROOT = PROJECT_ROOT / "sweep_results_metrics"

EXPECTED_RATIOS = 9
EXPECTED_RECTANGULAR_ROWS = (
    len(MODELS)
    * len(ATTACKS)
    * EXPECTED_RATIOS
)


# ============================================================
# helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_sweep_file(model_key, attack):
    """
    Find the aggregate sweep JSON.

    The metrics directory also contains files ending in
    '__per_image.json'. Those must not be used here because this
    collector reads the aggregate values from the summary JSON.
    """

    prefix = MODELS[model_key]["prefix"]

    candidates = sorted(
        SWEEP_ROOT.glob(
            f"sweep_M_{prefix}_A_{attack}_*.json"
        )
    )

    candidates = [
        path
        for path in candidates
        if not path.name.endswith(
            "__per_image.json"
        )
    ]

    if not candidates:
        print(
            "missing:",
            model_key,
            attack,
        )
        return None

    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple aggregate sweep files found for "
            f"{model_key} / {attack}:\n"
            + "\n".join(
                str(path)
                for path in candidates
            )
        )

    return candidates[0]


def require_metric(item, key, source):
    if key not in item:
        raise KeyError(
            f"Missing '{key}' in {source}. "
            "This collector expects the new metrics-format "
            "evaluation output."
        )

    value = item[key]

    if value is None:
        raise ValueError(
            f"Metric '{key}' is null in {source}."
        )

    return value


# ============================================================
# collect clean validation / embedding quality
# ============================================================

def collect_validation():
    """
    Clean validation contains no attack.

    Therefore PSNR/SSIM here describe embedding quality:
        cover -> encoded
    """

    rows = []

    for model, info in MODELS.items():

        data = load_json(
            info["validation"]
        )

        # Preserve compatibility with the old validation JSON
        # structure: the actual result is the first top-level value.
        result = list(data.values())[0]

        rows.append(
            {
                "model": model,
                "PSNR_embed": result["psnr"],
                "SSIM_embed": result["ssim"],
                "BER": result["ber"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# collect rectangular sweep
# ============================================================

def collect_rectangular():
    """
    Collect both image-quality definitions:

        final:
            cover -> attacked
            PSNR_final / SSIM_final

        attack-only:
            encoded -> attacked
            PSNR_attack / SSIM_attack

    BER is unchanged.
    """

    rows = []

    for model in MODELS:

        for attack in ATTACKS:

            path = find_sweep_file(
                model,
                attack,
            )

            if path is None:
                continue

            data = load_json(path)

            for ratio, item in data["results"].items():

                rows.append(
                    {
                        "model":
                            model,

                        "attack":
                            attack,

                        "removal_ratio":
                            item[
                                "target_removal_ratio"
                            ],

                        "actual_removal_ratio":
                            item.get(
                                "actual_removal_ratio"
                            ),

                        "PSNR_final":
                            require_metric(
                                item,
                                "psnr_final",
                                path,
                            ),

                        "SSIM_final":
                            require_metric(
                                item,
                                "ssim_final",
                                path,
                            ),

                        "PSNR_attack":
                            require_metric(
                                item,
                                "psnr_attack",
                                path,
                            ),

                        "SSIM_attack":
                            require_metric(
                                item,
                                "ssim_attack",
                                path,
                            ),

                        "BER":
                            require_metric(
                                item,
                                "ber",
                                path,
                            ),

                        "num_images":
                            data.get(
                                "num_images"
                            ),
                    }
                )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            [
                "model",
                "attack",
                "removal_ratio",
            ]
        ).reset_index(drop=True)

    return df


# ============================================================
# latex export
# ============================================================

def dataframe_to_latex(
    df,
    filename,
    caption,
    label,
):
    latex = df.to_latex(
        index=False,
        escape=True,
    )

    content = f"""
\\begin{{table}}[h]
\\centering

\\caption{{{caption}}}
\\label{{{label}}}

{latex}

\\end{{table}}
"""

    with open(
        TEX_DIR / filename,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(content)


# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    print("Collect validation...")

    val_df = collect_validation()

    val_df.to_csv(
        CSV_DIR / "validation_results.csv",
        index=False,
    )

    dataframe_to_latex(
        val_df,
        "validation_results.tex",
        "Clean validation performance. PSNR and SSIM measure embedding quality from cover to encoded image.",
        "tab:validation",
    )

    print(val_df)

    print(
        "\nCollect rectangular mask..."
    )

    rect_df = collect_rectangular()

    print(
        "Collected rectangular rows:",
        len(rect_df),
    )

    if len(rect_df) != EXPECTED_RECTANGULAR_ROWS:
        print(
            "WARNING: expected "
            f"{EXPECTED_RECTANGULAR_ROWS} rows!"
        )
    else:
        print(
            "Rectangular sanity check passed."
        )

    rect_df.to_csv(
        CSV_DIR
        / "rectangular_mask_results.csv",
        index=False,
    )

    dataframe_to_latex(
        rect_df,
        "rectangular_mask_results.tex",
        "Robustness evaluation under rectangular masks. Final image quality is measured against the cover image; attack-only quality is measured against the encoded image.",
        "tab:rectangular",
    )

    print(rect_df.head())
    print("\nDone.")
