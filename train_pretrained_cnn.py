import os
import csv
import time
import pickle
import random

import numpy as np
import torch
import torchvision

import utils

from options import (
    TrainingOptions,
    HiDDenConfiguration
)

from noise_layers.pretrained_cnn_inpainting import (
    ReconstructionCNN,
    RectangleMaskGenerator,
)


# =========================================================
# Reconstruction loss
# =========================================================

def masked_l1_loss(output, target, mask):
    """
    Compute reconstruction loss only inside the missing region.

    mask:
        1 = missing
        0 = known
    """

    error = torch.abs(
        output - target
    ) * mask

    denominator = (
        mask.sum() * target.shape[1]
        + 1e-8
    )

    return error.sum() / denominator


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    min_ratio,
    max_ratio,
    seed,
):

    model.eval()

    # Reset every epoch so validation uses exactly
    # the same masks each time.
    mask_generator = RectangleMaskGenerator(
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        seed=seed
    )

    total_loss = 0.0
    total_images = 0

    examples = None

    for image, _ in val_loader:

        image = image.to(
            device=device,
            dtype=torch.float32
        )

        B, C, H, W = image.shape

        mask = mask_generator.generate(
            B,
            H,
            W,
            device
        )

        reconstructed = model(
            image,
            mask
        )

        loss = masked_l1_loss(
            reconstructed,
            image,
            mask
        )

        total_loss += (
            loss.item() * B
        )

        total_images += B

        if examples is None:

            examples = (
                image.detach().cpu(),
                mask.detach().cpu(),
                reconstructed.detach().cpu()
            )

    return (
        total_loss / total_images,
        examples
    )


# =========================================================
# Save visualization
# =========================================================

def save_examples(
    examples,
    epoch,
    folder,
    n=8,
):

    original, mask, reconstructed = examples

    original = original[:n]
    mask = mask[:n]
    reconstructed = reconstructed[:n]

    # For visualization only:
    # zero in [-1,1] corresponds to grey.
    masked = original * (
        1.0 - mask
    )

    original = (
        original + 1.0
    ) / 2.0

    masked = (
        masked + 1.0
    ) / 2.0

    reconstructed = (
        reconstructed + 1.0
    ) / 2.0

    images = torch.cat(
        [
            original,
            masked,
            reconstructed
        ],
        dim=0
    )

    torchvision.utils.save_image(
        images,
        os.path.join(
            folder,
            'images',
            f'epoch-{epoch}.png'
        ),
        nrow=n
    )


# =========================================================
# Save checkpoint
# =========================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    file_name,
):

    torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        },
        file_name
    )


# =========================================================
# Main CNN training function
# =========================================================

def pretrain_cnn(
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
    runs_folder='./runs',
):

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -----------------------------------------------------
    # Reuse original HiDDeN data pipeline
    # -----------------------------------------------------

    train_options = TrainingOptions(
        batch_size=batch_size,
        number_of_epochs=epochs,
        train_folder=os.path.join(
            data_dir,
            'train'
        ),
        validation_folder=os.path.join(
            data_dir,
            'val'
        ),
        runs_folder=runs_folder,
        start_epoch=1,
        experiment_name=experiment_name
    )

    # Only H/W matter for get_data_loaders(),
    # but use the normal HiDDeN configuration
    # to keep everything consistent.
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

        enable_fp16=False
    )

    train_loader, val_loader = (
        utils.get_data_loaders(
            hidden_config,
            train_options
        )
    )

    # Original loader therefore still uses:
    #
    # train -> RandomCrop
    # val   -> CenterCrop
    # Normalize -> [-1, 1]

    # -----------------------------------------------------
    # Run folder
    # -----------------------------------------------------

    run_folder = utils.create_folder_for_run(
        runs_folder,
        experiment_name
    )

    config = {
        'experiment_name':
            experiment_name,

        'image_size':
            image_size,

        'batch_size':
            batch_size,

        'epochs':
            epochs,

        'learning_rate':
            lr,

        'min_mask_ratio':
            min_ratio,

        'max_mask_ratio':
            max_ratio,

        'seed':
            seed,

        'architecture':
            '4-32-32-32-3',

        'activation':
            'Tanh',

        'loss':
            'masked L1',

        'soft_mask':
            False,
    }

    with open(
        os.path.join(
            run_folder,
            'pretrain-config.pickle'
        ),
        'wb'
    ) as f:

        pickle.dump(
            config,
            f
        )

    print()
    print("=" * 70)
    print("CNN reconstruction pretraining")
    print("=" * 70)

    print("Device:", device)

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
        "Run folder:",
        run_folder
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = ReconstructionCNN(
        hidden_channels=32
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    train_mask_generator = (
        RectangleMaskGenerator(
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            seed=seed
        )
    )

    csv_file = os.path.join(
        run_folder,
        'train.csv'
    )

    with open(
        csv_file,
        'w',
        newline=''
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            'epoch',
            'train_l1',
            'val_l1',
            'duration'
        ])

    best_val_loss = float('inf')

    # =====================================================
    # Epoch loop
    # =====================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        start_time = time.time()

        model.train()

        total_loss = 0.0
        total_images = 0

        for image, _ in train_loader:

            image = image.to(
                device=device,
                dtype=torch.float32
            )

            B, C, H, W = image.shape

            mask = (
                train_mask_generator.generate(
                    B,
                    H,
                    W,
                    device
                )
            )

            reconstructed = model(
                image,
                mask
            )

            loss = masked_l1_loss(
                reconstructed,
                image,
                mask
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * B
            )

            total_images += B

        train_loss = (
            total_loss
            / total_images
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        val_loss, examples = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            min_ratio=min_ratio,
            max_ratio=max_ratio,

            # separate deterministic validation masks
            seed=seed + 1000
        )

        duration = (
            time.time()
            - start_time
        )

        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"| train L1 = "
            f"{train_loss:.6f} "
            f"| val L1 = "
            f"{val_loss:.6f} "
            f"| {duration:.1f}s"
        )

        # -------------------------------------------------
        # CSV
        # -------------------------------------------------

        with open(
            csv_file,
            'a',
            newline=''
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                duration
            ])

        # -------------------------------------------------
        # Last checkpoint
        # -------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            epoch,
            train_loss,
            val_loss,
            os.path.join(
                run_folder,
                'checkpoints',
                'last.pyt'
            )
        )

        # -------------------------------------------------
        # Best checkpoint
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                os.path.join(
                    run_folder,
                    'checkpoints',
                    'best.pyt'
                )
            )

            print(
                "  -> new best checkpoint"
            )

        # -------------------------------------------------
        # Visualization
        # -------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):

            save_examples(
                examples,
                epoch,
                run_folder
            )

    print()
    print("=" * 70)

    print(
        "CNN pretraining finished."
    )

    print(
        "Best validation L1:",
        best_val_loss
    )

    print(
        "Best checkpoint:",
        os.path.join(
            run_folder,
            'checkpoints',
            'best.pyt'
        )
    )

    print("=" * 70)