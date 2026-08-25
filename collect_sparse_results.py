import json
from pathlib import Path
import pandas as pd


# ============================================================
# paths
# ============================================================

OUTPUT = Path(
    "/home/keen/HiDDeN/output"
)

CSV_DIR = OUTPUT / "csv"
TEX_DIR = OUTPUT / "tex"

CSV_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OPT_ROOT = Path(
    "/home/keen/HiDDeN/optimized_mask_eval_outputs"
)

RANDOM_ROOT = Path(
    "/home/keen/HiDDeN/sparse_random_mask_eval_outputs"
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
        "3k_frozen_pretrained_unet_eval_aligned"

}


ATTACKS = [
    "mean",
    "telea",
    "navier",
    "diffusion"
]


DENSITIES = {
    "0.1": "100",
    "0.3": "300"
}


MASK_CONFIGS = [

    {
        "name": "Optimized_d150_n100",
        "type": "optimized",
        "diffusion": 150,
        "nlpe": 100
    },

    {
        "name": "Optimized_d300_n200",
        "type": "optimized",
        "diffusion": 300,
        "nlpe": 200
    },

    {
        "name": "Random",
        "type": "random"
    }

]



# ============================================================
# helpers
# ============================================================


def load_json(path):

    with open(path, "r") as f:
        return json.load(f)



def find_all_result_files():

    files=[]


    # optimized
    for f in OPT_ROOT.rglob(
        "evaluation_results.json"
    ):

        path=str(f)


        if "opt_masks_diff_" in path:

            files.append(
                f
            )


    # random
    for f in RANDOM_ROOT.rglob(
        "evaluation_results.json"
    ):

        path=str(f)


        if "random_masks_den_" in path:

            files.append(
                f
            )


    return files



# ============================================================
# collect
# ============================================================


def collect_sparse_results():


    rows=[]


    files=find_all_result_files()


    print(
        "Found files:",
        len(files)
    )


    for path in files:


        data=load_json(path)


        folder=str(path.parent)



        # -----------------------
        # model
        # -----------------------

        model_raw=data["model_name"]


        model_map={

            "3k_baseline":
                "Baseline",

            "3k_gaussian_blur":
                "Gaussian Blur",

            "3k_haar_wavelet":
                "Haar Wavelet",

            "3k_frozen_pretrained_unet_eval_aligned":
                "Frozen UNet"

        }


        model=model_map.get(
            model_raw,
            model_raw
        )



        # -----------------------
        # attack
        # -----------------------

        attack=data["attack"]



        # -----------------------
        # mask type
        # -----------------------

        if "opt_masks_diff_150_nlpe_100" in folder:

            mask_type="Optimized_d150_n100"


        elif "opt_masks_diff_300_nlpe_200" in folder:

            mask_type="Optimized_d300_n200"


        elif "random_masks_den_" in folder:

            mask_type="Random"


        else:

            continue



        # -----------------------
        # density
        # -----------------------

        if "density_100" in folder:

            density=0.1

        elif "density_300" in folder:

            density=0.3

        else:

            print(
                "unknown density",
                folder
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

            "PSNR":
                data["psnr"],

            "SSIM":
                data["ssim"],

            "BER":
                data["ber"],

            "num_images":
                data["processed_images"]

            }
        )


    return pd.DataFrame(rows)



# ============================================================
# latex
# ============================================================


def save_latex(df):


    latex = df.to_latex(
        index=False,
        escape=True
    )


    content = f"""
\\begin{{table}}[h]
\\centering

\\caption{{Sparse mask robustness evaluation.}}

\\label{{tab:sparse_mask_results}}

{latex}

\\end{{table}}
"""


    with open(
        TEX_DIR / "sparse_mask_results.tex",
        "w"
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
        len(df)
    )



    if len(df) != 96:

        print(
            "WARNING: expected 96 rows!"
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
                "density"
            ]
        ).size()
    )


    df.to_csv(
        CSV_DIR / "sparse_mask_results.csv",
        index=False
    )


    save_latex(df)


    print()
    print(
        "Saved:"
    )

    print(
        CSV_DIR / "sparse_mask_results.csv"
    )

    print(
        TEX_DIR / "sparse_mask_results.tex"
    )