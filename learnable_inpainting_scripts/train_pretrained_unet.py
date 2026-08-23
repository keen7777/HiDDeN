import os
import csv
import time
import pickle
import random

import numpy as np
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torchvision

import utils

from options import (
    TrainingOptions,
    HiDDenConfiguration,
)

from noise_layers.pretrained_unet_inpainting import (
    SmallUNet,
    ControlledRectangleMaskGenerator,
)


# ============================================================
# Reconstruction losses
# ============================================================

def masked_losses(
    output,
    target,
    mask,
    mse_weight=0.5,
):

    channels = target.shape[1]

    denominator = (
        mask.sum()
        * channels
        + 1e-8
    )

    error = (
        output - target
    )

    l1 = (
        torch.abs(error)
        * mask
    ).sum() / denominator

    mse = (
        (error ** 2)
        * mask
    ).sum() / denominator

    total = (
        l1
        + mse_weight * mse
    )

    return total, l1, mse


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    min_ratio,
    max_ratio,
    seed,
    mse_weight,
):

    model.eval()

    # Recreate generator every validation epoch:
    # exact same validation masks every time.
    mask_generator = (
        ControlledRectangleMaskGenerator(
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            seed=seed,
            randomize_ratio=True,
        )
    )

    total_loss = 0.0
    total_l1 = 0.0
    total_mse = 0.0
    total_images = 0

    examples = None

    actual_coverages = []

    for image, _ in val_loader:

        image = image.to(
            device=device,
            dtype=torch.float32,
        )

        B, C, H, W = image.shape

        mask = mask_generator.generate(
            B,
            H,
            W,
            device,
        )

        reconstructed = model(
            image,
            mask,
        )

        loss, l1, mse = masked_losses(
            reconstructed,
            image,
            mask,
            mse_weight=mse_weight,
        )

        total_loss += (
            loss.item() * B
        )

        total_l1 += (
            l1.item() * B
        )

        total_mse += (
            mse.item() * B
        )

        total_images += B

        actual_coverages.extend(
            mask.mean(
                dim=(1, 2, 3)
            ).cpu().tolist()
        )

        if examples is None:

            examples = (
                image.detach().cpu(),
                mask.detach().cpu(),
                reconstructed.detach().cpu(),
            )

    return {
        "loss":
            total_loss
            / total_images,

        "l1":
            total_l1
            / total_images,

        "mse":
            total_mse
            / total_images,

        "coverage":
            float(
                np.mean(
                    actual_coverages
                )
            ),

        "examples":
            examples,
    }


# ============================================================
# Visualization
# ============================================================

def save_examples(
    examples,
    epoch,
    folder,
    n=8,
):

    (
        original,
        mask,
        reconstructed,
    ) = examples

    n = min(
        n,
        original.shape[0],
    )

    original = original[:n]
    mask = mask[:n]
    reconstructed = reconstructed[:n]

    masked = (
        original
        * (1.0 - mask)
    )

    def vis(x):
        return torch.clamp(
            (x + 1.0) / 2.0,
            0.0,
            1.0,
        )

    images = torch.cat(
        [
            vis(original),
            vis(masked),
            vis(reconstructed),
        ],
        dim=0,
    )

    torchvision.utils.save_image(
        images,
        os.path.join(
            folder,
            "images",
            f"epoch-{epoch}.png",
        ),
        nrow=n,
    )


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    train_l1,
    train_mse,
    validation,
    config,
    filename,
):

    checkpoint = {
        "epoch":
            epoch,

        "model":
            model.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "train_l1":
            train_l1,

        "train_mse":
            train_mse,

        "val_loss":
            validation["loss"],

        "val_l1":
            validation["l1"],

        "val_mse":
            validation["mse"],

        "config":
            config,
    }

    torch.save(
        checkpoint,
        filename,
    )


# ============================================================
# Main training function
# ============================================================

def pretrain_unet(
    device,
    data_dir,
    batch_size,
    epochs,
    experiment_name,
    image_size,
    lr,
    min_ratio,
    max_ratio,
    seed,
    mse_weight=0.5,
    runs_folder="./runs",
):

    # ========================================================
    # Reproducibility
    # ========================================================

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(
            "Warning: deterministic algorithms "
            "could not be fully enabled:",
            e
        )

    # ========================================================
    # Reuse original HiDDeN data pipeline
    # ========================================================

    train_options = TrainingOptions(
        batch_size=batch_size,
        number_of_epochs=epochs,

        train_folder=os.path.join(
            data_dir,
            "train",
        ),

        validation_folder=os.path.join(
            data_dir,
            "val",
        ),

        runs_folder=runs_folder,
        start_epoch=1,
        experiment_name=experiment_name,
    )

    hidden_config = HiDDenConfiguration(
        H=image_size,
        W=image_size,

        message_length=30,

        encoder_blocks=4,
        encoder_channels=64,

        decoder_blocks=7,
        decoder_channels=64,

        use_discriminator=True,
        use_vgg=False,

        discriminator_blocks=3,
        discriminator_channels=64,

        decoder_loss=1,
        encoder_loss=0.7,
        adversarial_loss=1e-3,

        enable_fp16=False,
    )

    train_loader, val_loader = (
        utils.get_data_loaders(
            hidden_config,
            train_options,
        )
    )

    # ========================================================
    # Run folder
    # ========================================================

    run_folder = (
        utils.create_folder_for_run(
            runs_folder,
            experiment_name,
        )
    )

    config = {
        "model":
            "SmallUNet",

        "base_channels":
            32,

        "image_size":
            image_size,

        "batch_size":
            batch_size,

        "epochs":
            epochs,

        "learning_rate":
            lr,

        "min_mask_ratio":
            min_ratio,

        "max_mask_ratio":
            max_ratio,

        "controlled_coverage":
            True,

        "soft_mask":
            False,

        "loss":
            "masked_L1 + mse_weight * masked_MSE",

        "mse_weight":
            mse_weight,

        "seed":
            seed,
    }

    with open(
        os.path.join(
            run_folder,
            "pretrain-config.pickle",
        ),
        "wb",
    ) as f:

        pickle.dump(
            config,
            f,
        )

    print()
    print("=" * 70)
    print(
        "U-Net reconstruction pretraining"
    )
    print("=" * 70)

    print(
        "Device:",
        device
    )

    print(
        "Train images:",
        len(train_loader.dataset)
    )

    print(
        "Validation images:",
        len(val_loader.dataset)
    )

    print(
        "Image size:",
        image_size
    )

    print(
        "Mask ratio:",
        min_ratio,
        "-",
        max_ratio
    )

    print(
        "MSE weight:",
        mse_weight
    )

    print(
        "Run folder:",
        run_folder
    )

    # ========================================================
    # Model
    # ========================================================

    model = SmallUNet(
        base_channels=32
    ).to(device)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    train_mask_generator = (
        ControlledRectangleMaskGenerator(
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            seed=seed,
            randomize_ratio=True,
        )
    )

    # ========================================================
    # CSV
    # ========================================================

    csv_file = os.path.join(
        run_folder,
        "train.csv",
    )

    with open(
        csv_file,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_l1",
            "train_mse",
            "val_loss",
            "val_l1",
            "val_mse",
            "val_actual_coverage",
            "duration",
        ])

    best_val_loss = float("inf")

    # ========================================================
    # Epochs
    # ========================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        epoch_start = time.time()

        model.train()

        total_loss = 0.0
        total_l1 = 0.0
        total_mse = 0.0
        total_images = 0

        for image, _ in train_loader:

            image = image.to(
                device=device,
                dtype=torch.float32,
            )

            B, C, H, W = image.shape

            mask = (
                train_mask_generator.generate(
                    B,
                    H,
                    W,
                    device,
                )
            )

            reconstructed = model(
                image,
                mask,
            )

            loss, l1, mse = masked_losses(
                reconstructed,
                image,
                mask,
                mse_weight=mse_weight,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * B
            )

            total_l1 += (
                l1.item() * B
            )

            total_mse += (
                mse.item() * B
            )

            total_images += B

        train_loss = (
            total_loss
            / total_images
        )

        train_l1 = (
            total_l1
            / total_images
        )

        train_mse = (
            total_mse
            / total_images
        )

        # ====================================================
        # Validation
        # ====================================================

        validation = validate(
            model=model,
            val_loader=val_loader,
            device=device,

            min_ratio=min_ratio,
            max_ratio=max_ratio,

            # Fixed validation masks.
            seed=seed + 1000,

            mse_weight=mse_weight,
        )

        duration = (
            time.time()
            - epoch_start
        )

        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"| train "
            f"L={train_loss:.5f} "
            f"L1={train_l1:.5f} "
            f"MSE={train_mse:.5f} "
            f"| val "
            f"L={validation['loss']:.5f} "
            f"L1={validation['l1']:.5f} "
            f"MSE={validation['mse']:.5f} "
            f"| coverage="
            f"{validation['coverage']:.4f} "
            f"| {duration:.1f}s"
        )

        # ====================================================
        # CSV
        # ====================================================

        with open(
            csv_file,
            "a",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                train_l1,
                train_mse,

                validation["loss"],
                validation["l1"],
                validation["mse"],
                validation["coverage"],

                duration,
            ])

        # ====================================================
        # Last checkpoint
        # ====================================================

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,

            train_loss=train_loss,
            train_l1=train_l1,
            train_mse=train_mse,

            validation=validation,
            config=config,

            filename=os.path.join(
                run_folder,
                "checkpoints",
                "last.pyt",
            ),
        )

        # ====================================================
        # Best checkpoint
        # ====================================================

        if (
            validation["loss"]
            < best_val_loss
        ):

            best_val_loss = (
                validation["loss"]
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,

                train_loss=train_loss,
                train_l1=train_l1,
                train_mse=train_mse,

                validation=validation,
                config=config,

                filename=os.path.join(
                    run_folder,
                    "checkpoints",
                    "best.pyt",
                ),
            )

            print(
                "  -> new best checkpoint"
            )

        # ====================================================
        # Visual examples
        # ====================================================

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):

            save_examples(
                validation["examples"],
                epoch,
                run_folder,
            )

    print()
    print("=" * 70)
    print(
        "U-Net pretraining finished."
    )

    print(
        "Best validation loss:",
        best_val_loss
    )

    print(
        "Best checkpoint:",
        os.path.join(
            run_folder,
            "checkpoints",
            "best.pyt",
        )
    )

    print("=" * 70)