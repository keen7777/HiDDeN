import sys
import json
import subprocess
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_SCRIPT = (
    PROJECT_ROOT
    / "optimization_mask_files"
    / "evaluate_optimized_masks.py"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "optimized_mask_eval_outputs"
)

DATA_DIR = PROJECT_ROOT / "images"
RUNS_ROOT = PROJECT_ROOT / "runs"


# ============================================================
# Models
# ============================================================

MODELS = [
    "3k_baseline 2026.05.13--12-17-40",
    "3k_gaussian_blur 2026.05.26--18-42-22",
    "3k_haar_wavelet 2026.05.26--12-12-55",
    "3k_frozen_pretrained_unet 2026.08.22--12-55-32",
]


# ============================================================
# Optimized mask sets
#
# The diffusion iterations used during evaluation are matched
# to the diffusion setting used when the mask set was created:
#
#   diff150 masks -> evaluate with DiffusionFill(150)
#   diff300 masks -> evaluate with DiffusionFill(300)
#
# All four final mask sets use mask-generation seed 42.
# ============================================================

MASK_SETS = [
    {
        "path": (
            PROJECT_ROOT
            / "opt_masks_data"
            / "opt_masks_diff_150_nlpe_100_den_0.1"
            / "val_ps_nlpe_masks_density_100.npy"
        ),
        "diffusion_iterations": 150,
    },
    {
        "path": (
            PROJECT_ROOT
            / "opt_masks_data"
            / "opt_masks_diff_150_nlpe_100_den_0.3"
            / "val_ps_nlpe_masks_density_300.npy"
        ),
        "diffusion_iterations": 150,
    },
    {
        "path": (
            PROJECT_ROOT
            / "opt_masks_data"
            / "opt_masks_diff_300_nlpe_200_den_0.1"
            / "val_ps_nlpe_masks_density_100.npy"
        ),
        "diffusion_iterations": 300,
    },
    {
        "path": (
            PROJECT_ROOT
            / "opt_masks_data"
            / "opt_masks_diff_300_nlpe_200_den_0.3"
            / "val_ps_nlpe_masks_density_300.npy"
        ),
        "diffusion_iterations": 300,
    },
]


# ============================================================
# Reconstruction methods
# ============================================================

# First finish the matched optimized-mask + diffusion experiment.
"""
ATTACKS = [
    "diffusion",
]
"""
ATTACKS = [
    "diffusion",
]

ATTACKS = [
    "diffusion",
    "mean",
    "telea",
    "navier",
]


# ============================================================
# Evaluation settings
# ============================================================

MESSAGE_SEED = 42
BATCH_SIZE = 1

DIFFUSION_TAU = 0.25

# Save the same first N validation examples for every run,
# which makes visual comparisons across models/mask sets easy.
SAVE_IMAGES = True
SAVE_MASK_VISUALS = True
SAVE_IMAGE_LIMIT = 10


# ============================================================
# Helpers
# ============================================================

def get_output_dir(
    model,
    attack,
    mask_file,
):
    """
    Must match the naming logic in
    evaluate_optimized_masks.py exactly.
    """

    model_name = model.split(" ")[0]

    mask_stem = mask_file.stem
    mask_set_name = mask_file.parent.name

    dirname = (
        f"M_{model_name}"
        f"__A_{attack}"
        f"__{mask_set_name}"
        f"__{mask_stem}"
        f"__seed_{MESSAGE_SEED}"
    )

    return OUTPUT_ROOT / dirname


def result_is_complete(
    result_file,
    model,
    attack,
    mask_file,
    diffusion_iterations,
):
    """
    Skip only if the stored result really matches the current
    experiment settings.

    When image saving is enabled, an old result with no saved
    visualizations is deliberately treated as incomplete so that
    it is rerun and the requested images are produced.
    """

    if not result_file.exists():
        return False

    try:
        with open(
            result_file,
            "r",
            encoding="utf-8",
        ) as f:
            result = json.load(f)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return False

    required_fields = [
        "run_name",
        "attack",
        "mask_file",
        "processed_images",
        "psnr",
        "ssim",
        "ber",
    ]

    if not all(
        field in result
        for field in required_fields
    ):
        return False

    if result["run_name"] != model:
        return False

    if result["attack"] != attack:
        return False

    try:
        saved_mask = Path(
            result["mask_file"]
        ).resolve()

        expected_mask = (
            mask_file.resolve()
        )

        if saved_mask != expected_mask:
            return False

    except Exception:
        return False

    if result["processed_images"] != 300:
        return False

    # Check the matched diffusion reconstruction setting.
    if attack == "diffusion":

        if (
            result.get(
                "diffusion_iterations"
            )
            != diffusion_iterations
        ):
            return False

        tau = result.get("tau")

        if tau is None:
            return False

        if abs(
            float(tau)
            - DIFFUSION_TAU
        ) > 1e-12:
            return False

    # If we currently request images, do not skip an older
    # numerical-only run that saved no visualizations.
    if SAVE_IMAGES:

        expected_visualizations = (
            300
            if SAVE_IMAGE_LIMIT == -1
            else min(
                SAVE_IMAGE_LIMIT,
                300,
            )
        )

        if (
            int(
                result.get(
                    "saved_visualizations",
                    0,
                )
            )
            < expected_visualizations
        ):
            return False

    return True


def check_inputs():
    """
    Fail immediately if a required path is missing.
    """

    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Evaluation script not found:\n"
            f"{EVAL_SCRIPT}"
        )

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{DATA_DIR}"
        )

    if not RUNS_ROOT.exists():
        raise FileNotFoundError(
            f"Runs directory not found:\n"
            f"{RUNS_ROOT}"
        )

    for mask_set in MASK_SETS:

        mask_file = mask_set["path"]

        if not mask_file.exists():
            raise FileNotFoundError(
                f"Mask file not found:\n"
                f"{mask_file}"
            )

    for model in MODELS:

        run_dir = (
            RUNS_ROOT
            / model
        )

        if not run_dir.exists():
            raise FileNotFoundError(
                f"Model run not found:\n"
                f"{run_dir}"
            )


# ============================================================
# Main
# ============================================================

def main():

    check_inputs()

    total_runs = (
        len(MODELS)
        * len(MASK_SETS)
        * len(ATTACKS)
    )

    print("=" * 70)
    print("OPTIMIZED MASK EVALUATION GRID")
    print("=" * 70)
    print(
        f"Models      : {len(MODELS)}"
    )
    print(
        f"Mask sets   : {len(MASK_SETS)}"
    )
    print(
        f"Attacks     : {len(ATTACKS)}"
    )
    print(
        f"Total runs  : {total_runs}"
    )
    print(
        f"Message seed: {MESSAGE_SEED}"
    )
    print(
        f"Save images : {SAVE_IMAGES}"
    )

    if SAVE_IMAGES:
        print(
            f"Image limit : {SAVE_IMAGE_LIMIT}"
        )
        print(
            f"Save masks  : {SAVE_MASK_VISUALS}"
        )

    print("=" * 70)

    completed = 0
    skipped = 0
    executed = 0

    for model in MODELS:

        model_name = (
            model.split(" ")[0]
        )

        for mask_set in MASK_SETS:

            mask_file = (
                mask_set["path"]
            )

            diffusion_iterations = (
                mask_set[
                    "diffusion_iterations"
                ]
            )

            mask_set_name = (
                mask_file.parent.name
            )

            for attack in ATTACKS:

                completed += 1

                output_dir = get_output_dir(
                    model=model,
                    attack=attack,
                    mask_file=mask_file,
                )

                result_file = (
                    output_dir
                    / "evaluation_results.json"
                )

                print()
                print("=" * 70)
                print(
                    f"[{completed}/{total_runs}]"
                )
                print(
                    f"MODEL : {model_name}"
                )
                print(
                    f"MASK  : {mask_set_name}"
                )
                print(
                    f"ATTACK: {attack}"
                )

                if attack == "diffusion":
                    print(
                        "EVAL DIFFUSION ITERATIONS: "
                        f"{diffusion_iterations}"
                    )
                    print(
                        f"EVAL TAU: {DIFFUSION_TAU}"
                    )

                print("=" * 70)

                # --------------------------------------------
                # Resume support
                # --------------------------------------------

                if result_is_complete(
                    result_file=result_file,
                    model=model,
                    attack=attack,
                    mask_file=mask_file,
                    diffusion_iterations=diffusion_iterations,
                ):

                    print(
                        "SKIP: completed matching result "
                        "already exists:"
                    )

                    print(
                        result_file
                    )

                    skipped += 1
                    continue

                # --------------------------------------------
                # Command
                # --------------------------------------------

                cmd = [
                    sys.executable,
                    str(EVAL_SCRIPT),

                    "-d",
                    str(DATA_DIR),

                    "-r",
                    str(RUNS_ROOT),

                    "--run-name",
                    model,

                    "--mask-file",
                    str(mask_file),

                    "--attack",
                    attack,

                    "--batch-size",
                    str(BATCH_SIZE),

                    "--message-seed",
                    str(MESSAGE_SEED),

                    "--output-root",
                    str(OUTPUT_ROOT),
                ]

                # Use the reconstruction setting matched to
                # the mask-generation diffusion budget.
                if attack == "diffusion":

                    cmd.extend(
                        [
                            "--tau",
                            str(
                                DIFFUSION_TAU
                            ),

                            "--diffusion-iterations",
                            str(
                                diffusion_iterations
                            ),
                        ]
                    )

                # --------------------------------------------
                # Optional images
                # --------------------------------------------

                if SAVE_IMAGES:

                    cmd.append(
                        "--save-images"
                    )

                    cmd.extend(
                        [
                            "--save-image-limit",
                            str(
                                SAVE_IMAGE_LIMIT
                            ),
                        ]
                    )

                    if SAVE_MASK_VISUALS:

                        cmd.append(
                            "--save-mask-visuals"
                        )

                # --------------------------------------------
                # Run
                # --------------------------------------------

                print(
                    "Starting evaluation..."
                )

                subprocess.run(
                    cmd,
                    check=True,
                    cwd=PROJECT_ROOT,
                )

                executed += 1

                print(
                    "DONE:"
                )

                print(
                    result_file
                )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 70)
    print("GRID FINISHED")
    print("=" * 70)

    print(
        f"Total combinations : "
        f"{total_runs}"
    )

    print(
        f"Executed now       : "
        f"{executed}"
    )

    print(
        f"Skipped completed  : "
        f"{skipped}"
    )

    print(
        f"Output root        : "
        f"{OUTPUT_ROOT}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
