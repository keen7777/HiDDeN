import os
import pprint
import argparse
import torch
import pickle
import utils
import logging
import sys

from options import *
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from noise_argparser import NoiseArgParser
from train import train


def main():
    # device 设置
    # device = torch.device("cpu")  # 默认 CPU，可改成 "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # argparse 配置
    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')

    # 新训练 run
    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser.add_argument('--data-dir', '-d', required=True, type=str, help='Data directory.')
    new_run_parser.add_argument('--batch-size', '-b', required=True, type=int, help='Batch size.')
    new_run_parser.add_argument('--epochs', '-e', default=300, type=int, help='Number of epochs.')
    new_run_parser.add_argument('--name', required=True, type=str, help='Experiment name.')
    new_run_parser.add_argument('--size', '-s', default=128, type=int, help='Image size (H=W).')
    new_run_parser.add_argument('--message', '-m', default=30, type=int, help='Message length in bits.')
    new_run_parser.add_argument('--continue-from-folder', '-c', default='', type=str,
                                help='Folder to continue previous run.')
    new_run_parser.add_argument('--tensorboard', action='store_true', help='Enable Tensorboard logging.')
    new_run_parser.add_argument('--enable-fp16', action='store_true', help='Enable mixed-precision training.')
    new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                                help="Noise layers configuration. Use quotes, e.g., 'cropout((0.55,0.6),(0.55,0.6))'")

    new_run_parser.set_defaults(tensorboard=False)
    new_run_parser.set_defaults(enable_fp16=False)

    # 继续训练 run
    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    continue_parser.add_argument('--folder', '-f', required=True, type=str, help='Checkpoint folder.')
    continue_parser.add_argument('--data-dir', '-d', required=False, type=str, help='Optional override data directory.')
    continue_parser.add_argument('--epochs', '-e', required=False, type=int, help='Optional override number of epochs.')

    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_file_name = None

    # 判断新训练还是继续
    if args.command == 'continue':
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(this_run_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch'] + 1

        if args.data_dir is not None:
            train_options.train_folder = os.path.join(args.data_dir, 'train')
            train_options.validation_folder = os.path.join(args.data_dir, 'val')

        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(f'Command-line specifies epochs={args.epochs}, but folder={this_run_folder} '
                      f'already has checkpoint for epoch={train_options.start_epoch}.')
                exit(1)
    else:
        assert args.command == 'new'
        start_epoch = 1
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, 'train'),
            validation_folder=os.path.join(args.data_dir, 'val'),
            runs_folder=os.path.join('.', 'runs'),
            start_epoch=start_epoch,
            experiment_name=args.name
        )

        noise_config = args.noise if args.noise is not None else []
        hidden_config = HiDDenConfiguration(
            H=args.size, W=args.size,
            message_length=args.message,
            encoder_blocks=4, encoder_channels=64,
            decoder_blocks=7, decoder_channels=64,
            use_discriminator=True,
            use_vgg=False,
            discriminator_blocks=3, discriminator_channels=64,
            decoder_loss=1,
            encoder_loss=0.7,
            adversarial_loss=1e-3,
            enable_fp16=args.enable_fp16
        )

        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(noise_config, f)
            pickle.dump(hidden_config, f)

    # 日志配置
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])

    if (args.command == 'new' and args.tensorboard) or \
       (args.command == 'continue' and os.path.isdir(os.path.join(this_run_folder, 'tb-logs'))):
        logging.info('Tensorboard enabled. Creating logger.')
        from tensorboard_logger import TensorBoardLogger
        tb_logger = TensorBoardLogger(os.path.join(this_run_folder, 'tb-logs'))
    else:
        tb_logger = None

    # 初始化模型
    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, tb_logger)

    # 继续训练时加载 checkpoint
    if args.command == 'continue':
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(model, checkpoint)

    # 打印模型信息
    logging.info('HiDDeN model: {}\n'.format(str(model)))
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(hidden_config)))
    logging.info('\nNoise configuration:\n')
    logging.info(pprint.pformat(str(noise_config)))
    logging.info('\nTraining options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    # 开始训练
    train(model, device, hidden_config, train_options, this_run_folder, tb_logger)


if __name__ == '__main__':
    main()