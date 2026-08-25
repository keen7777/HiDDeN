import json
from pathlib import Path
import pandas as pd


# ============================================================
# paths
# ============================================================

OUTPUT = Path("/home/keen/HiDDeN/output")

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
        "/home/keen/HiDDeN/evaluation_3k_baseline_results.json"
    },

    "Gaussian Blur": {
        "validation":
        "/home/keen/HiDDeN/evaluation_3k_gaussian_blur_results.json"
    },

    "Haar Wavelet": {
        "validation":
        "/home/keen/HiDDeN/evaluation_3k_haar_wavelet_results.json"
    },

    "Frozen UNet": {
        "validation":
        "/home/keen/HiDDeN/evaluation_3k_frozen_pretrained_unet_eval_aligned_results.json"
    }
}


# ============================================================
# rectangular sweep
# ============================================================

ATTACKS = [
    "mean",
    "telea",
    "navier",
    "shiftmap",
    "diffusion"
]


SWEEP_ROOT = Path(
    "/home/keen/HiDDeN/sweep_results"
)


def load_json(path):

    with open(path,"r") as f:
        return json.load(f)



def find_sweep_file(model_key, attack):

    """
    match filenames automatically
    """

    name_map = {

        "Baseline":
        "3k_baseline",

        "Gaussian Blur":
        "3k_gaussian_blur",

        "Haar Wavelet":
        "3k_haar_wavelet",

        "Frozen UNet":
        "3k_frozen_pretrained_unet_eval_aligned"
    }


    prefix = name_map[model_key]


    files = list(
        SWEEP_ROOT.glob(
            f"sweep_M_{prefix}_A_{attack}_*.json"
        )
    )


    if len(files)==0:
        print(
            "missing:",
            model_key,
            attack
        )
        return None


    return files[0]



# ============================================================
# collect validation
# ============================================================


def collect_validation():

    rows=[]


    for model,info in MODELS.items():

        data = load_json(
            info["validation"]
        )

        # first key
        result=list(data.values())[0]


        rows.append(
            {
                "model":model,
                "PSNR":result["psnr"],
                "SSIM":result["ssim"],
                "BER":result["ber"]
            }
        )


    return pd.DataFrame(rows)



# ============================================================
# collect rectangular mask
# ============================================================


def collect_rectangular():

    rows=[]


    for model in MODELS:


        for attack in ATTACKS:


            path=find_sweep_file(
                model,
                attack
            )


            if path is None:
                continue


            data=load_json(path)


            for ratio,item in data["results"].items():

                rows.append(
                    {
                        "model":model,
                        "attack":attack,
                        "removal_ratio":
                            item["target_removal_ratio"],

                        "PSNR":
                            item["psnr"],

                        "SSIM":
                            item["ssim"],

                        "BER":
                            item["ber"]
                    }
                )


    return pd.DataFrame(rows)



# ============================================================
# latex export
# ============================================================


def dataframe_to_latex(
        df,
        filename,
        caption,
        label
):

    latex = df.to_latex(
        index=False,
        escape=True
    )


    content=f"""
\\begin{{table}}[h]
\\centering

\\caption{{{caption}}}
\\label{{{label}}}

{latex}

\\end{{table}}
"""


    with open(
        TEX_DIR/filename,
        "w"
    ) as f:

        f.write(content)



# ============================================================
# main
# ============================================================


if __name__=="__main__":


    print("Collect validation...")

    val_df=collect_validation()

    val_df.to_csv(
        CSV_DIR/"validation_results.csv",
        index=False
    )


    dataframe_to_latex(
        val_df,
        "validation_results.tex",
        "Validation performance before attacks.",
        "tab:validation"
    )


    print(val_df)



    print("\nCollect rectangular mask...")


    rect_df=collect_rectangular()


    rect_df.to_csv(
        CSV_DIR/"rectangular_mask_results.csv",
        index=False
    )


    dataframe_to_latex(
        rect_df,
        "rectangular_mask_results.tex",
        "Robustness evaluation under rectangular masks.",
        "tab:rectangular"
    )


    print(rect_df.head())

    print("\nDone.")