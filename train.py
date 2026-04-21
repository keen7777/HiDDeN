import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict

from options import *
from model.hidden import Hidden
from average_meter import AverageMeter


def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually GPU if available, otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object. Pass None to disable logging.
    :return:
    """

    train_data, val_data = utils.get_data_loaders(hidden_config, train_options)
    file_count = len(train_data.dataset)
    steps_in_epoch = (file_count + train_options.batch_size - 1) // train_options.batch_size

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info(f'\nStarting epoch {epoch}/{train_options.number_of_epochs}')
        logging.info(f'Batch size = {train_options.batch_size}\nSteps in epoch = {steps_in_epoch}')
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1

        for image, _ in train_data:
            # 确保 tensor 类型 float32，并放在 device 上
            image = image.to(device=device, dtype=torch.float32)
            message = torch.tensor(
                np.random.choice([0, 1], (image.shape[0], hidden_config.message_length)),
                dtype=torch.float32,
                device=device
            )

            # 训练单 batch
            losses, _ = model.train_on_batch([image, message])

            # 更新 loss，确保用 .item()
            for name, loss in losses.items():
                training_losses[name].update(loss)

            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(f'Epoch: {epoch}/{train_options.number_of_epochs} Step: {step}/{steps_in_epoch}')
                utils.log_progress(training_losses)
                logging.info('-' * 40)

            step += 1

        train_duration = time.time() - epoch_start
        logging.info(f'Epoch {epoch} training duration {train_duration:.2f} sec')
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)

        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        # --- Validation ---
        first_iteration = True
        validation_losses = defaultdict(AverageMeter)
        logging.info(f'Running validation for epoch {epoch}/{train_options.number_of_epochs}')

        for image, _ in val_data:
            image = image.to(device=device, dtype=torch.float32)
            message = torch.tensor(
                np.random.choice([0, 1], (image.shape[0], hidden_config.message_length)),
                dtype=torch.float32,
                device=device
            )

            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message])

            for name, loss in losses.items():
                validation_losses[name].update(loss)

            if first_iteration:
                # 保存示例图片
                utils.save_images(
                    image.cpu()[:images_to_save],
                    encoded_images[:images_to_save].cpu(),
                    epoch,
                    os.path.join(this_run_folder, 'images'),
                    resize_to=saved_images_size
                )
                first_iteration = False

        utils.log_progress(validation_losses)
        logging.info('-' * 40)

        # 保存 checkpoint
        utils.save_checkpoint(
            model,
            train_options.experiment_name,
            epoch,
            os.path.join(this_run_folder, 'checkpoints')
        )

        utils.write_losses(
            os.path.join(this_run_folder, 'validation.csv'),
            validation_losses,
            epoch,
            time.time() - epoch_start
        )