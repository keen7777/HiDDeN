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
    / "sparse_random_mask_files"
    / "evaluate_sparse_random_masks.py"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "sparse_random_mask_eval_outputs"
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
    "3k_frozen_pretrained_unet_eval_aligned 2026.08.24--04-07-29",
]


# ============================================================
# Random sparse mask sets
#
# Masks are generated with fixed seeds.
#
# mask value:
#   1 -> retained pixel
#   0 -> removed pixel
#
# density means retained pixel ratio.
# ============================================================

MASK_SETS = [
    {
        "path":
            PROJECT_ROOT
            / "sparse_random_masks_data"
            / "random_masks_den_0.1_seed_42"
            / "val_random_masks_density_100.npy",

        "density": 0.1,
        "seed": 42,
    },


    {
        "path":
            PROJECT_ROOT
            / "sparse_random_masks_data"
            / "random_masks_den_0.3_seed_42"
            / "val_random_masks_density_300.npy",

        "density": 0.3,
        "seed": 42,
    },

]


# ============================================================
# Reconstruction methods
# ============================================================

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

DIFFUSION_ITERATIONS = 150


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

    model_name = model.split(" ")[0]

    mask_set_name = mask_file.parent.name

    mask_stem = mask_file.stem


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
):

    if not result_file.exists():
        return False


    try:

        with open(
            result_file,
            "r",
            encoding="utf-8",
        ) as f:

            result = json.load(f)


    except Exception:

        return False



    required = [
        "run_name",
        "attack",
        "mask_file",
        "processed_images",
        "psnr",
        "ssim",
        "ber",
    ]


    if not all(
        x in result
        for x in required
    ):
        return False



    if result["run_name"] != model:
        return False


    if result["attack"] != attack:
        return False



    try:

        if (
            Path(result["mask_file"]).resolve()
            != mask_file.resolve()
        ):
            return False

    except Exception:

        return False



    if result["processed_images"] != 300:

        return False



    if SAVE_IMAGES:

        expected = min(
            SAVE_IMAGE_LIMIT,
            300,
        )

        if (
            result.get(
                "saved_visualizations",
                0,
            )
            < expected
        ):
            return False


    return True




def check_inputs():

    if not EVAL_SCRIPT.exists():

        raise FileNotFoundError(
            EVAL_SCRIPT
        )


    if not DATA_DIR.exists():

        raise FileNotFoundError(
            DATA_DIR
        )


    if not RUNS_ROOT.exists():

        raise FileNotFoundError(
            RUNS_ROOT
        )



    for mask in MASK_SETS:

        if not mask["path"].exists():

            raise FileNotFoundError(
                mask["path"]
            )



# ============================================================
# Main
# ============================================================


def main():


    check_inputs()



    total_runs = (
        len(MODELS)
        *
        len(MASK_SETS)
        *
        len(ATTACKS)
    )



    print("=" * 70)

    print(
        "RANDOM SPARSE MASK EVALUATION GRID"
    )

    print("=" * 70)


    print(
        f"Models     : {len(MODELS)}"
    )

    print(
        f"Mask sets  : {len(MASK_SETS)}"
    )

    print(
        f"Attacks    : {len(ATTACKS)}"
    )

    print(
        f"Total runs : {total_runs}"
    )

    print("=" * 70)




    counter = 0

    executed = 0

    skipped = 0



    for model in MODELS:


        for mask_set in MASK_SETS:


            mask_file = mask_set["path"]



            for attack in ATTACKS:


                counter += 1



                output_dir = get_output_dir(
                    model,
                    attack,
                    mask_file,
                )


                result_file = (
                    output_dir
                    /
                    "evaluation_results.json"
                )



                print()
                print("=" * 70)

                print(
                    f"[{counter}/{total_runs}]"
                )

                print(
                    "MODEL:",
                    model
                )

                print(
                    "MASK:",
                    mask_file.parent.name
                )

                print(
                    "ATTACK:",
                    attack
                )

                print("=" * 70)



                if result_is_complete(
                    result_file,
                    model,
                    attack,
                    mask_file,
                ):

                    print(
                        "SKIP existing result"
                    )

                    skipped += 1

                    continue



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



                if attack == "diffusion":

                    cmd.extend(
                        [
                            "--tau",
                            str(DIFFUSION_TAU),

                            "--diffusion-iterations",
                            str(
                                DIFFUSION_ITERATIONS
                            ),
                        ]
                    )



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



                subprocess.run(
                    cmd,
                    check=True,
                    cwd=PROJECT_ROOT,
                )


                executed += 1



    print()
    print("=" * 70)

    print(
        "GRID FINISHED"
    )

    print("=" * 70)

    print(
        "Executed:",
        executed
    )

    print(
        "Skipped:",
        skipped
    )


if __name__ == "__main__":

    main()