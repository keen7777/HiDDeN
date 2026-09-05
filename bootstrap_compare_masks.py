import json
from pathlib import Path

import numpy as np


# ============================================================
# Input files
# ============================================================

RANDOM_FILE = Path(
    "/home/keen/HiDDeN/"
    "sparse_random_mask_eval_outputs_metrics/"
    "M_3k_baseline__A_diffusion__random_masks_den_0.1_seed_42"
    "__val_random_masks_density_100__seed_42/"
    "per_image_metrics.json"
)

O150_FILE = Path(
    "/home/keen/HiDDeN/"
    "optimized_mask_eval_outputs_metrics/"
    "M_3k_baseline__A_diffusion__opt_masks_diff_150_nlpe_100_den_0.1"
    "__val_ps_nlpe_masks_density_100__seed_42/"
    "per_image_metrics.json"
)

O300_FILE = Path(
    "/home/keen/HiDDeN/"
    "optimized_mask_eval_outputs_metrics/"
    "M_3k_baseline__A_diffusion__opt_masks_diff_300_nlpe_200_den_0.1"
    "__val_ps_nlpe_masks_density_100__seed_42/"
    "per_image_metrics.json"
)


# ============================================================
# Bootstrap settings
# ============================================================

N_BOOTSTRAP = 10_000
SEED = 42

# These are the metrics that are most useful for the thesis.
METRICS = [
    "ber",
    "psnr_attack",
    "ssim_attack",
    "psnr_final",
    "ssim_final",
]


# ============================================================
# Helpers
# ============================================================

def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a list in {path}, "
            f"got {type(records).__name__}"
        )

    return records


def records_by_index(records):
    result = {}

    for record in records:
        index = int(record["index"])

        if index in result:
            raise ValueError(
                f"Duplicate image index: {index}"
            )

        result[index] = record

    return result


def get_metric(record, metric):
    """
    Compatibility helper.

    New files contain 'ber'.
    Older files may only contain 'ber_batch_value'.
    """

    if metric == "ber":
        if "ber" in record:
            return float(record["ber"])

        return float(record["ber_batch_value"])

    return float(record[metric])


def paired_bootstrap(
    records_a,
    records_b,
    metric,
    n_bootstrap=10_000,
    seed=42,
):
    """
    Compare B against A.

    delta = B - A
    """

    by_index_a = records_by_index(records_a)
    by_index_b = records_by_index(records_b)

    indices_a = set(by_index_a)
    indices_b = set(by_index_b)

    if indices_a != indices_b:
        missing_in_a = sorted(indices_b - indices_a)
        missing_in_b = sorted(indices_a - indices_b)

        raise ValueError(
            "Image indices do not match.\n"
            f"Missing in A: {missing_in_a[:10]}\n"
            f"Missing in B: {missing_in_b[:10]}"
        )

    indices = sorted(indices_a)

    values_a = np.array(
        [
            get_metric(
                by_index_a[index],
                metric,
            )
            for index in indices
        ],
        dtype=np.float64,
    )

    values_b = np.array(
        [
            get_metric(
                by_index_b[index],
                metric,
            )
            for index in indices
        ],
        dtype=np.float64,
    )

    # Paired image-level differences.
    differences = values_b - values_a

    n_images = len(differences)

    rng = np.random.default_rng(seed)

    # Each bootstrap sample resamples IMAGE INDICES,
    # preserving the pairing between the two conditions.
    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for bootstrap_index in range(n_bootstrap):
        sampled_indices = rng.integers(
            low=0,
            high=n_images,
            size=n_images,
        )

        bootstrap_means[bootstrap_index] = (
            differences[sampled_indices].mean()
        )

    ci_low, ci_high = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    return {
        "n_images": int(n_images),

        "mean_a": float(values_a.mean()),
        "mean_b": float(values_b.mean()),

        "mean_difference": float(
            differences.mean()
        ),

        "sd_difference": float(
            differences.std(ddof=1)
        ),

        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),

        "bootstrap_samples": int(n_bootstrap),
    }


def print_result(
    comparison_name,
    metric,
    result,
):
    print("-" * 70)

    print(
        f"{comparison_name} | {metric}"
    )

    print(
        f"N = {result['n_images']}"
    )

    print(
        f"Mean A = "
        f"{result['mean_a']:.6f}"
    )

    print(
        f"Mean B = "
        f"{result['mean_b']:.6f}"
    )

    print(
        f"Delta (B - A) = "
        f"{result['mean_difference']:+.6f}"
    )

    print(
        "95% bootstrap CI = "
        f"[{result['ci_95_low']:+.6f}, "
        f"{result['ci_95_high']:+.6f}]"
    )

    if (
        result["ci_95_low"] > 0
        or result["ci_95_high"] < 0
    ):
        print(
            "CI excludes zero."
        )
    else:
        print(
            "CI includes zero."
        )


# ============================================================
# Main
# ============================================================

def main():

    random_records = load_records(
        RANDOM_FILE
    )

    o150_records = load_records(
        O150_FILE
    )

    o300_records = load_records(
        O300_FILE
    )

    comparisons = [
        (
            "Random -> O150",
            random_records,
            o150_records,
        ),
        (
            "Random -> O300",
            random_records,
            o300_records,
        ),
        (
            "O150 -> O300",
            o150_records,
            o300_records,
        ),
    ]

    all_results = {}

    print("=" * 70)
    print(
        "PAIRED IMAGE-LEVEL BOOTSTRAP"
    )
    print("=" * 70)
    print(
        f"Bootstrap samples: {N_BOOTSTRAP}"
    )
    print(
        f"Seed: {SEED}"
    )

    for (
        comparison_name,
        records_a,
        records_b,
    ) in comparisons:

        all_results[
            comparison_name
        ] = {}

        print()
        print("=" * 70)
        print(comparison_name)
        print("=" * 70)

        for metric in METRICS:

            result = paired_bootstrap(
                records_a=records_a,
                records_b=records_b,
                metric=metric,
                n_bootstrap=N_BOOTSTRAP,
                seed=SEED,
            )

            all_results[
                comparison_name
            ][metric] = result

            print_result(
                comparison_name,
                metric,
                result,
            )

    # --------------------------------------------------------
    # Save complete numerical output
    # --------------------------------------------------------

    output_file = Path(
        "bootstrap_baseline_diffusion_d01.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print(
        f"Saved -> {output_file.resolve()}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()