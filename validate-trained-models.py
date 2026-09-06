import os
import time
import pprint
import argparse
import torch
import numpy as np
import pickle
import utils
import csv
import json
import random  # Keen: reproducible evaluation

from model.hidden import Hidden
from noise_layers.noiser import Noiser
from average_meter import AverageMeter
from evaluation.rgb_yuv_convert import convert_img_range, rgb_to_yuv
from evaluation.psnr_ssim_evaluation import compute_psnr, compute_ssim


def write_validation_loss(file_name, losses_accu, experiment_name, epoch, write_header=False):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            row_to_write = ['experiment_name', 'epoch'] + [
                loss_name.strip() for loss_name in losses_accu.keys()
            ]
            writer.writerow(row_to_write)

        row_to_write = [experiment_name, epoch] + [
            '{:.4f}'.format(loss_avg.avg)
            for loss_avg in losses_accu.values()
        ]
        writer.writerow(row_to_write)


def main():
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Training of HiDDeN nets')

    # parser.add_argument(
    #     '--size',
    #     '-s',
    #     default=128,
    #     type=int,
    #     help='The size of the images '
    #          '(images are square so this is height and width).'
    # )

    parser.add_argument(
        '--data-dir',
        '-d',
        required=True,
        type=str,
        help='The directory where the data is stored.'
    )

    parser.add_argument(
        '--runs_root',
        '-r',
        default=os.path.join('.', 'experiments'),
        type=str,
        help='The root folder where data about experiments are stored.'
    )

    parser.add_argument(
        '--batch-size',
        '-b',
        default=1,
        type=int,
        help='Validation batch size.'
    )

    # Keen:
    # only run 1 model:
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Run only one specific model.'
    )

    # ============================================================
    # Keen: reproducible evaluation
    # ------------------------------------------------------------
    # Use a fixed random seed so that different trained models are
    # evaluated with the same watermark messages and, as far as
    # possible, the same random state for stochastic noise layers.
    #
    # This only affects validation/evaluation. It does NOT retrain
    # or modify the stored model checkpoints.
    # ============================================================
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed used for reproducible validation.'
    )

    args = parser.parse_args()
    print_each = 100

    # Keen:
    # If run-name is specified, validate only one model
    if args.run_name is not None:
        completed_runs = [args.run_name]
    else:
        completed_runs = [
            o for o in os.listdir(args.runs_root)
            if os.path.isdir(os.path.join(args.runs_root, o))
            and o != 'no-noise-defaults'
        ]

    # completed_runs = [
    #     o for o in os.listdir(args.runs_root)
    #     if os.path.isdir(os.path.join(args.runs_root, o))
    #     and o != 'no-noise-defaults'
    # ]

    print(completed_runs)

    # ============================================================
    # Keen:
    # Only write the CSV header when the output file does not yet
    # exist or is empty. This prevents repeated headers when the
    # validation script is run multiple times.
    # ============================================================
    validation_csv = os.path.join(
        args.runs_root,
        'validation_run.csv'
    )

    write_csv_header = (
        not os.path.exists(validation_csv)
        or os.path.getsize(validation_csv) == 0
    )

    # Keen:
    # adding output json for all the results, later graph:
    all_results = {}

    for run_name in completed_runs:
        current_run = os.path.join(args.runs_root, run_name)
        print(f'Run folder: {current_run}')

        options_file = os.path.join(
            current_run,
            'options-and-config.pickle'
        )

        train_options, hidden_config, noise_config = \
            utils.load_options(options_file)

        train_options.train_folder = os.path.join(
            args.data_dir,
            'val'
        )

        train_options.validation_folder = os.path.join(
            args.data_dir,
            'val'
        )

        train_options.batch_size = args.batch_size

        checkpoint, chpt_file_name = utils.load_last_checkpoint(
            os.path.join(current_run, 'checkpoints')
        )

        print(f'Loaded checkpoint from file {chpt_file_name}')

        # Keen:
        # adding modified para: device
        noiser = Noiser(noise_config, device)

        model = Hidden(
            hidden_config,
            device,
            noiser,
            tb_logger=None
        )

        utils.model_from_checkpoint(model, checkpoint)

        # ========================================================
        # Keen: reproducible evaluation
        # --------------------------------------------------------
        # Reset all commonly used random number generators for
        # EACH model.
        #
        # Therefore every model starts validation from the same
        # random state.
        #
        # This is useful for stochastic mask/noise layers using:
        #   - Python random
        #   - NumPy random
        #   - PyTorch random
        #
        # Note:
        # The watermark messages themselves are generated below
        # using a separate torch.Generator. Therefore random calls
        # made inside a noise layer cannot change the messages.
        # ========================================================
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print(
            'Model loaded successfully. '
            'Starting validation run...'
        )

        _, val_data = utils.get_data_loaders(
            hidden_config,
            train_options
        )

        file_count = len(val_data.dataset)

        # ========================================================
        # Keen: reproducible watermark messages
        # --------------------------------------------------------
        # Generate the complete set of binary messages ONCE before
        # validation.
        #
        # Because the generator uses the same seed for every model,
        # validation image i receives the same message for:
        #
        #   baseline
        #   resize
        #   dropout
        #   CNN inpainting
        #   Gaussian
        #   wavelet
        #   ...
        #
        # This removes message sampling as a source of variation
        # when comparing BER across models.
        #
        # The dedicated Generator is deliberately independent from
        # the global PyTorch RNG used by stochastic noise layers.
        # ========================================================
        message_generator = torch.Generator(device='cpu')
        message_generator.manual_seed(args.seed)

        fixed_messages = torch.randint(
            low=0,
            high=2,
            size=(
                file_count,
                hidden_config.message_length
            ),
            generator=message_generator,
            dtype=torch.float32
        )

        print(
            f'Using fixed validation messages '
            f'(seed={args.seed}, images={file_count}).'
        )

        if file_count % train_options.batch_size == 0:
            steps_in_epoch = (
                file_count // train_options.batch_size
            )
        else:
            steps_in_epoch = (
                file_count // train_options.batch_size + 1
            )

        losses_accu = {}

        # Keen:
        # adding psnr rgb mean:
        psnr_rgb_meter = AverageMeter()

        # adding psnr yuv mean:
        psnr_y_meter = AverageMeter()
        psnr_u_meter = AverageMeter()
        psnr_v_meter = AverageMeter()

        # adding ssim mean:
        ssim_meter = AverageMeter()

        step = 0

        # ========================================================
        # Keen: reproducible evaluation
        # Keep track of the current position in fixed_messages.
        # This also works when validation batch size > 1.
        # ========================================================
        message_offset = 0

        for image, _ in val_data:
            step += 1

            image = image.to(device)

            # ====================================================
            # Keen: reproducible evaluation
            # ----------------------------------------------------
            # OLD:
            #
            # message = torch.Tensor(
            #     np.random.choice(
            #         [0, 1],
            #         (
            #             image.shape[0],
            #             hidden_config.message_length
            #         )
            #     )
            # ).to(device)
            #
            # NEW:
            # Select the corresponding pre-generated message(s).
            # ====================================================
            current_batch_size = image.shape[0]

            message = fixed_messages[
                message_offset:
                message_offset + current_batch_size
            ].to(device)

            message_offset += current_batch_size

            # losses, (
            #     encoded_images,
            #     noised_images,
            #     decoded_messages
            # ) = model.validate_on_batch(
            #     [image, message],
            #     set_eval_mode=True
            # )

            # Keen:
            # Keep the original validation behaviour unchanged.
            losses, (
                encoded_images,
                noised_images,
                decoded_messages
            ) = model.validate_on_batch(
                [image, message]
            )

            # Keen:
            # adding psnr rgb mean:
            #
            # IMPORTANT:
            # These PSNR/SSIM values measure the visual quality of
            # the ENCODED watermarked image relative to the cover:
            #
            #   cover image <-> encoded image
            #
            # They do NOT measure:
            #
            #   cover image <-> noised/attacked image
            #
            # Attack image quality should be evaluated separately
            # in the sweep/evaluation scripts.
            cover = convert_img_range(image)
            encoded = convert_img_range(encoded_images)

            psnr = compute_psnr(
                cover,
                encoded
            )

            psnr_rgb_meter.update(
                psnr.item()
            )

            # SSIM for human eye
            ssim = compute_ssim(
                cover,
                encoded
            )

            ssim_meter.update(
                ssim.item()
            )

            # RGB -> YUV
            # since human eyes are more sensitive to Y channel
            cover_yuv = rgb_to_yuv(cover)
            encoded_yuv = rgb_to_yuv(encoded)

            cover_y, cover_u, cover_v = cover_yuv
            encoded_y, encoded_u, encoded_v = encoded_yuv

            psnr_y = compute_psnr(
                cover_y,
                encoded_y
            )

            psnr_y_meter.update(
                psnr_y.item()
            )

            psnr_u = compute_psnr(
                cover_u,
                encoded_u
            )

            psnr_u_meter.update(
                psnr_u.item()
            )

            psnr_v = compute_psnr(
                cover_v,
                encoded_v
            )

            psnr_v_meter.update(
                psnr_v.item()
            )

            if not losses_accu:
                # dict is empty, initialize
                for name in losses:
                    losses_accu[name] = AverageMeter()

            for name, loss in losses.items():
                losses_accu[name].update(loss)

            if (
                step % print_each == 0
                or step == steps_in_epoch
            ):
                print(
                    f'Step {step}/{steps_in_epoch}'
                )
                utils.print_progress(losses_accu)
                print('-' * 40)

        # utils.print_progress(losses_accu)

        # Keen:
        # print out psnr

        # ============================
        # ADD: final dataset-level metrics
        # ============================
        print("====================================")
        print(
            f"[{run_name}] "
            f"DATASET EVALUATION RESULTS"
        )
        print("------------------------------------")
        print(
            f"Evaluation seed: {args.seed}"
        )
        print(
            f"Number of images: {file_count}"
        )
        print(
            f"Mean PSNR: "
            f"{psnr_rgb_meter.avg:.4f}"
        )
        print(
            f"Mean PSNR(Y): "
            f"{psnr_y_meter.avg:.4f}"
        )
        print(
            f"Mean PSNR(U): "
            f"{psnr_u_meter.avg:.4f}"
        )
        print(
            f"Mean PSNR(V): "
            f"{psnr_v_meter.avg:.4f}"
        )
        print(
            f"SSIM: "
            f"{ssim_meter.avg:.4f}"
        )
        print(
            f"Mean BER : "
            f"{losses_accu['bitwise-error  '].avg:.6f}"
        )
        print("====================================")

        # Save metrics into dictionary
        model_name = run_name.split(" ")[0]

        all_results[model_name] = {
            # Keen:
            # Save evaluation settings together with metrics
            # so the experiment can be reproduced later.
            "seed": args.seed,
            "num_images": file_count,
            "checkpoint_epoch": checkpoint['epoch'],

            "psnr": round(
                psnr_rgb_meter.avg,
                4
            ),

            "psnr_y": round(
                psnr_y_meter.avg,
                4
            ),

            "psnr_u": round(
                psnr_u_meter.avg,
                4
            ),

            "psnr_v": round(
                psnr_v_meter.avg,
                4
            ),

            "ssim": round(
                ssim_meter.avg,
                4
            ),

            "ber": round(
                losses_accu[
                    'bitwise-error  '
                ].avg,
                6
            )
        }

        write_validation_loss(
            validation_csv,
            losses_accu,
            run_name,
            checkpoint['epoch'],
            write_header=write_csv_header
        )

        # After the first result has been written, do not add
        # another header during the same script execution.
        write_csv_header = False

        with open(
            f"validate_trained_models_result/evaluation_{model_name}_results.json",
            "w"
        ) as f:
            json.dump(
                all_results,
                f,
                indent=4
            )

        print(
            f"Saved evaluation {run_name} results to "
            f"evaluation_{model_name}_results.json"
        )

    # train(
    #     model,
    #     device,
    #     hidden_config,
    #     train_options,
    #     this_run_folder,
    #     tb_logger
    # )


if __name__ == '__main__':
    main()