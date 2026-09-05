import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODELS = [
    "3k_baseline",
    "3k_gaussian_blur",
    "3k_haar_wavelet",
    "3k_frozen_pretrained_unet_eval_aligned",
]

ATTACKS = [
    "diffusion",
    "mean",
    "telea",
    "navier",
]

DENSITIES = {
    0.1: 100,
    0.3: 300,
}

MASK_VARIANTS = {
    "Random": {"kind": "random"},
    "O150": {
        "kind": "optimized",
        "diff": 150,
        "nlpe": 100,
    },
    "O300": {
        "kind": "optimized",
        "diff": 300,
        "nlpe": 200,
    },
}

COMPARISONS = [
    ("Random", "O150"),
    ("Random", "O300"),
    ("O150", "O300"),
]

METRICS = [
    "ber",
    "psnr_attack",
    "ssim_attack",
    "psnr_final",
    "ssim_final",
]


def build_metric_path(
    project_root,
    model,
    attack,
    density,
    density_code,
    variant,
):
    spec = MASK_VARIANTS[variant]

    if spec["kind"] == "random":
        root = (
            project_root
            / "sparse_random_mask_eval_outputs_metrics"
        )

        dirname = (
            f"M_{model}"
            f"__A_{attack}"
            f"__random_masks_den_{density}_seed_42"
            f"__val_random_masks_density_{density_code}"
            f"__seed_42"
        )

    else:
        root = (
            project_root
            / "optimized_mask_eval_outputs_metrics"
        )

        dirname = (
            f"M_{model}"
            f"__A_{attack}"
            f"__opt_masks_diff_{spec['diff']}"
            f"_nlpe_{spec['nlpe']}"
            f"_den_{density}"
            f"__val_ps_nlpe_masks_density_{density_code}"
            f"__seed_42"
        )

    return (
        root
        / dirname
        / "per_image_metrics.json"
    )


def build_mapping_path(
    project_root,
    density,
    density_code,
    variant,
):
    """
    Build the mask-index -> source-image mapping path.

    These files are generated together with the masks and are
    independent of model / attack.
    """

    spec = MASK_VARIANTS[variant]

    if spec["kind"] == "random":
        return (
            project_root
            / "sparse_random_masks_data"
            / f"random_masks_den_{density}_seed_42"
            / f"val_mask_mapping_density_{density_code}.json"
        )

    return (
        project_root
        / "opt_masks_data"
        / (
            f"opt_masks_diff_{spec['diff']}"
            f"_nlpe_{spec['nlpe']}"
            f"_den_{density}"
        )
        / f"val_mask_mapping_density_{density_code}.json"
    )


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected list in {path}"
        )

    by_index = {}

    for record in records:
        index = int(record["index"])

        if index in by_index:
            raise ValueError(
                f"Duplicate index {index} in {path}"
            )

        by_index[index] = record

    return by_index


def normalize_image_id(image_id):
    """
    Canonicalize image identifiers across mapping formats.

    Examples:
        "000000000139"
        "000000000139.jpg"
        "val_class/000000000139.jpg"

    all become:
        "000000000139"

    COCO validation image stems are unique, so comparing the stem
    is sufficient for verifying Random / O150 / O300 alignment.
    """
    image_id = str(image_id).replace("\\", "/")
    return Path(image_id).stem


def load_mapping(path):
    """
    Normalize one mask mapping to:

        {mask_index: image_id}

    Prefer 'image_id'; fall back to 'image_name' for older files.
    """

    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if not isinstance(mapping, list):
        raise ValueError(
            f"Expected list in mapping file: {path}"
        )

    by_index = {}

    for position, record in enumerate(mapping):
        if not isinstance(record, dict):
            raise ValueError(
                f"Mapping entry {position} in {path} "
                "is not an object."
            )

        if "mask_index" in record:
            mask_index = int(record["mask_index"])
        elif "index" in record:
            mask_index = int(record["index"])
        else:
            raise KeyError(
                f"Mapping entry {position} in {path} "
                "has neither 'mask_index' nor 'index'."
            )

        if "image_id" in record:
            image_id = normalize_image_id(
                record["image_id"]
            )
        elif "image_name" in record:
            image_id = normalize_image_id(
                record["image_name"]
            )
        else:
            raise KeyError(
                f"Mapping entry {position} in {path} "
                "has neither 'image_id' nor 'image_name'."
            )

        if mask_index in by_index:
            raise ValueError(
                f"Duplicate mask index {mask_index} in {path}"
            )

        by_index[mask_index] = image_id

    return by_index


def validate_mapping_alignment(
    mappings,
    expected_n_images,
):
    """
    Verify that Random / O150 / O300 map each mask index to the
    same validation image.

    Any mismatch is a hard error because paired bootstrap would
    otherwise pair different images.
    """

    variants = list(mappings)

    if not variants:
        raise ValueError("No mappings supplied.")

    reference_variant = variants[0]
    reference = mappings[reference_variant]
    reference_indices = set(reference)

    expected_indices = set(
        range(expected_n_images)
    )

    if len(reference) != expected_n_images:
        raise ValueError(
            f"{reference_variant} mapping has "
            f"{len(reference)} entries; expected "
            f"{expected_n_images}."
        )

    if reference_indices != expected_indices:
        missing = sorted(
            expected_indices - reference_indices
        )
        extra = sorted(
            reference_indices - expected_indices
        )

        raise ValueError(
            f"{reference_variant} mapping indices are not "
            f"exactly 0..{expected_n_images - 1}. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )

    for variant in variants[1:]:
        current = mappings[variant]

        if len(current) != expected_n_images:
            raise ValueError(
                f"{variant} mapping has "
                f"{len(current)} entries; expected "
                f"{expected_n_images}."
            )

        current_indices = set(current)

        if current_indices != reference_indices:
            missing = sorted(
                reference_indices - current_indices
            )
            extra = sorted(
                current_indices - reference_indices
            )

            raise ValueError(
                f"Mapping indices differ for {variant}. "
                f"Missing={missing[:10]}, extra={extra[:10]}"
            )

        mismatches = []

        for index in sorted(reference_indices):
            if current[index] != reference[index]:
                mismatches.append(
                    (
                        index,
                        reference[index],
                        current[index],
                    )
                )

                if len(mismatches) >= 10:
                    break

        if mismatches:
            details = "\n".join(
                (
                    f"index={index}: "
                    f"{reference_variant}={ref_id!r}, "
                    f"{variant}={cur_id!r}"
                )
                for index, ref_id, cur_id
                in mismatches
            )

            raise ValueError(
                "Mask-to-image mappings are not aligned.\n"
                f"{details}"
            )

    return sorted(reference_indices)


def validate_metric_indices_against_mapping(
    records,
    mapping_indices,
    variant,
):
    """
    Verify that per-image metrics cover exactly the same indices
    as the corresponding mask mapping.
    """

    metric_indices = set(records)
    expected_indices = set(mapping_indices)

    if metric_indices != expected_indices:
        missing = sorted(
            expected_indices - metric_indices
        )
        extra = sorted(
            metric_indices - expected_indices
        )

        raise ValueError(
            f"{variant} per-image metric indices do not match "
            f"the mask mapping. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )


def get_metric(record, metric):
    if metric == "ber":
        if "ber" in record:
            return float(record["ber"])

        return float(record["ber_batch_value"])

    return float(record[metric])


def aligned_metric_matrix(records_a, records_b):
    indices_a = set(records_a)
    indices_b = set(records_b)

    if indices_a != indices_b:
        raise ValueError(
            "Image indices do not match."
        )

    indices = sorted(indices_a)

    values_a = np.array(
        [
            [
                get_metric(
                    records_a[index],
                    metric,
                )
                for metric in METRICS
            ]
            for index in indices
        ],
        dtype=np.float64,
    )

    values_b = np.array(
        [
            [
                get_metric(
                    records_b[index],
                    metric,
                )
                for metric in METRICS
            ]
            for index in indices
        ],
        dtype=np.float64,
    )

    return values_a, values_b


def make_bootstrap_counts(
    n_images,
    n_bootstrap,
    seed,
):
    rng = np.random.default_rng(seed)

    probabilities = np.full(
        n_images,
        1.0 / n_images,
        dtype=np.float64,
    )

    return rng.multinomial(
        n_images,
        probabilities,
        size=n_bootstrap,
    )


def paired_bootstrap_all_metrics(
    values_a,
    values_b,
    bootstrap_counts,
):
    differences = values_b - values_a

    n_images = differences.shape[0]

    bootstrap_means = (
        bootstrap_counts @ differences
    ) / n_images

    ci_low = np.percentile(
        bootstrap_means,
        2.5,
        axis=0,
    )

    ci_high = np.percentile(
        bootstrap_means,
        97.5,
        axis=0,
    )

    results = {}

    for metric_index, metric in enumerate(
        METRICS
    ):
        low = float(ci_low[metric_index])
        high = float(ci_high[metric_index])

        results[metric] = {
            "n_images": int(n_images),
            "mean_a": float(
                values_a[:, metric_index].mean()
            ),
            "mean_b": float(
                values_b[:, metric_index].mean()
            ),
            "mean_difference": float(
                differences[:, metric_index].mean()
            ),
            "sd_difference": float(
                differences[
                    :,
                    metric_index,
                ].std(ddof=1)
            ),
            "ci_95_low": low,
            "ci_95_high": high,
            "ci_excludes_zero": bool(
                low > 0
                or high < 0
            ),
        }

    return results


def read_eval_metadata(per_image_path):
    summary_path = (
        per_image_path.parent
        / "evaluation_results.json"
    )

    if not summary_path.exists():
        return {}

    try:
        with open(
            summary_path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/keen/HiDDeN"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--require-all",
        action="store_true",
    )

    args = parser.parse_args()

    project_root = args.project_root.resolve()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            project_root
            / "bootstrap_results_metrics"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_n_images = 300

    bootstrap_counts = make_bootstrap_counts(
        expected_n_images,
        args.bootstrap_samples,
        args.seed,
    )

    nested_results = {}
    csv_rows = []
    missing_files = []
    warnings = []

    total_conditions = (
        len(MODELS)
        * len(ATTACKS)
        * len(DENSITIES)
    )

    counter = 0

    for model in MODELS:
        nested_results[model] = {}

        for attack in ATTACKS:
            nested_results[model][attack] = {}

            for density, density_code in DENSITIES.items():
                counter += 1

                print(
                    f"[{counter}/{total_conditions}] "
                    f"{model} | {attack} | d={density}"
                )

                paths = {
                    variant: build_metric_path(
                        project_root,
                        model,
                        attack,
                        density,
                        density_code,
                        variant,
                    )
                    for variant in MASK_VARIANTS
                }

                mapping_paths = {
                    variant: build_mapping_path(
                        project_root,
                        density,
                        density_code,
                        variant,
                    )
                    for variant in MASK_VARIANTS
                }

                missing_here = [
                    (
                        variant,
                        "metrics",
                        path,
                    )
                    for variant, path in paths.items()
                    if not path.exists()
                ]

                missing_here.extend(
                    [
                        (
                            variant,
                            "mapping",
                            path,
                        )
                        for variant, path
                        in mapping_paths.items()
                        if not path.exists()
                    ]
                )

                if missing_here:
                    print("  SKIP: missing input(s)")

                    for (
                        variant,
                        input_type,
                        path,
                    ) in missing_here:
                        print(
                            f"    {variant} "
                            f"[{input_type}]: {path}"
                        )

                        missing_files.append(
                            {
                                "model": model,
                                "attack": attack,
                                "density": density,
                                "variant": variant,
                                "input_type": input_type,
                                "path": str(path),
                            }
                        )

                    if args.require_all:
                        raise FileNotFoundError(
                            str(missing_here[0][2])
                        )

                    continue

                # --------------------------------------------
                # Verify mask -> image alignment BEFORE paired
                # bootstrap.
                # --------------------------------------------

                mappings = {
                    variant: load_mapping(path)
                    for variant, path
                    in mapping_paths.items()
                }

                mapping_indices = (
                    validate_mapping_alignment(
                        mappings,
                        expected_n_images,
                    )
                )

                print(
                    "  Mapping check: PASS "
                    f"({len(mapping_indices)} "
                    "aligned images)"
                )

                records = {
                    variant: load_records(path)
                    for variant, path in paths.items()
                }

                for variant in MASK_VARIANTS:
                    validate_metric_indices_against_mapping(
                        records[
                            variant
                        ],
                        mapping_indices,
                        variant,
                    )

                n_images = len(
                    next(iter(records.values()))
                )

                if n_images != expected_n_images:
                    raise ValueError(
                        f"Expected {expected_n_images} "
                        f"images, got {n_images}."
                    )

                metadata = {
                    variant: read_eval_metadata(path)
                    for variant, path in paths.items()
                }

                if attack == "diffusion":
                    diffusion_iters = {
                        variant: metadata[
                            variant
                        ].get(
                            "diffusion_iterations"
                        )
                        for variant in MASK_VARIANTS
                    }

                    known_iters = [
                        value
                        for value in diffusion_iters.values()
                        if value is not None
                    ]

                    if (
                        known_iters
                        and len(set(known_iters)) > 1
                    ):
                        warnings.append(
                            {
                                "model": model,
                                "attack": attack,
                                "density": density,
                                "diffusion_iterations": diffusion_iters,
                                "message": (
                                    "Diffusion iterations differ "
                                    "across variants; this is not "
                                    "a pure mask-only comparison."
                                ),
                            }
                        )

                        print(
                            "  WARNING diffusion iterations differ: "
                            f"{diffusion_iters}"
                        )

                condition_results = {}

                for variant_a, variant_b in COMPARISONS:
                    values_a, values_b = (
                        aligned_metric_matrix(
                            records[variant_a],
                            records[variant_b],
                        )
                    )

                    comparison_name = (
                        f"{variant_a} -> {variant_b}"
                    )

                    result = (
                        paired_bootstrap_all_metrics(
                            values_a,
                            values_b,
                            bootstrap_counts,
                        )
                    )

                    condition_results[
                        comparison_name
                    ] = result

                    for metric in METRICS:
                        r = result[metric]

                        csv_rows.append(
                            {
                                "model": model,
                                "attack": attack,
                                "density": density,
                                "comparison": comparison_name,
                                "metric": metric,
                                "n_images": r["n_images"],
                                "mean_a": r["mean_a"],
                                "mean_b": r["mean_b"],
                                "mean_difference": (
                                    r["mean_difference"]
                                ),
                                "ci_95_low": r["ci_95_low"],
                                "ci_95_high": r["ci_95_high"],
                                "ci_excludes_zero": (
                                    r["ci_excludes_zero"]
                                ),
                                "bootstrap_samples": (
                                    args.bootstrap_samples
                                ),
                                "bootstrap_seed": args.seed,
                            }
                        )

                    ber = result["ber"]
                    psnr = result["psnr_attack"]

                    print(
                        f"  {comparison_name:15s} "
                        f"| ΔBER {ber['mean_difference']:+.4f} "
                        f"[{ber['ci_95_low']:+.4f}, "
                        f"{ber['ci_95_high']:+.4f}] "
                        f"| ΔPSNR_attack "
                        f"{psnr['mean_difference']:+.3f} "
                        f"[{psnr['ci_95_low']:+.3f}, "
                        f"{psnr['ci_95_high']:+.3f}]"
                    )

                nested_results[
                    model
                ][attack][str(density)] = (
                    condition_results
                )

    json_output = (
        output_dir
        / "bootstrap_grid_results.json"
    )

    csv_output = (
        output_dir
        / "bootstrap_grid_summary.csv"
    )

    diagnostics_output = (
        output_dir
        / "bootstrap_grid_diagnostics.json"
    )

    with open(
        json_output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            nested_results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    fieldnames = [
        "model",
        "attack",
        "density",
        "comparison",
        "metric",
        "n_images",
        "mean_a",
        "mean_b",
        "mean_difference",
        "ci_95_low",
        "ci_95_high",
        "ci_excludes_zero",
        "bootstrap_samples",
        "bootstrap_seed",
    ]

    with open(
        csv_output,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(
        diagnostics_output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "missing_files": missing_files,
                "warnings": warnings,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("BOOTSTRAP GRID FINISHED")
    print(f"Rows written : {len(csv_rows)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Warnings     : {len(warnings)}")
    print(f"JSON         : {json_output}")
    print(f"CSV          : {csv_output}")
    print(f"Diagnostics  : {diagnostics_output}")


if __name__ == "__main__":
    main()
