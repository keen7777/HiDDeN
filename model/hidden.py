import numpy as np
import torch
import torch.nn as nn

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser


class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger=None):
        """
        HiDDeN model wrapper, includes encoder-decoder, discriminator, loss functions, and optimizer setup.
        :param configuration: HiDDenConfiguration object
        :param device: torch.device object
        :param noiser: Noiser object representing stacked noise layers
        :param tb_logger: optional TensorBoardLogger object
        """
        self.device = device
        self.config = configuration
        self.tb_logger = tb_logger

        # Encoder-Decoder and Discriminator
        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.discriminator = Discriminator(configuration).to(device)

        # Optimizers
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters(), lr=1e-4)
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        # Optional VGG perceptual loss
        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False).to(device)
        else:
            self.vgg_loss = None

        # Loss functions
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        # Discriminator labels
        self.cover_label = 1
        self.encoded_label = 0

        # TensorBoard hooks
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            # safe access to named modules
            encoder_final = dict(self.encoder_decoder.encoder.named_modules()).get('final_layer', None)
            if encoder_final is not None:
                encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = dict(self.encoder_decoder.decoder.named_modules()).get('linear', None)
            if decoder_final is not None:
                decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            discrim_final = dict(self.discriminator.named_modules()).get('linear', None)
            if discrim_final is not None:
                discrim_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))

    def train_on_batch(self, batch: list):
        images, messages = batch
        batch_size = images.shape[0]

        self.encoder_decoder.train()
        self.discriminator.train()

        # Ensure messages are float tensor on correct device
        messages = messages.float().to(self.device)

        # ---------------- Train Discriminator ----------------
        self.optimizer_discrim.zero_grad()
        d_target_cover = torch.full((batch_size, 1), self.cover_label, dtype=torch.float32, device=self.device)
        d_target_encoded = torch.full((batch_size, 1), self.encoded_label, dtype=torch.float32, device=self.device)
        g_target_encoded = torch.full((batch_size, 1), self.cover_label, dtype=torch.float32, device=self.device)

        # on cover images
        d_on_cover = self.discriminator(images)
        d_loss_cover = self.bce_with_logits_loss(d_on_cover, d_target_cover)
        d_loss_cover.backward()

        # on encoded images
        encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
        d_on_encoded = self.discriminator(encoded_images.detach())
        d_loss_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_encoded)
        d_loss_encoded.backward()
        self.optimizer_discrim.step()

        # ---------------- Train Encoder-Decoder ----------------
        self.optimizer_enc_dec.zero_grad()
        d_on_encoded_for_enc = self.discriminator(encoded_images)
        g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_encoded)

        if self.vgg_loss is None:
            g_loss_enc = self.mse_loss(encoded_images, images)
        else:
            vgg_on_cover = self.vgg_loss(images)
            vgg_on_encoded = self.vgg_loss(encoded_images)
            g_loss_enc = self.mse_loss(vgg_on_cover, vgg_on_encoded)

        g_loss_dec = self.mse_loss(decoded_messages, messages)
        g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                 + self.config.decoder_loss * g_loss_dec

        g_loss.backward()
        self.optimizer_enc_dec.step()

        # Bitwise error
        decoded_rounded = decoded_messages.detach().cpu().float().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().float().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_cover.item(),
            'discr_encod_bce': d_loss_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        images, messages = batch
        batch_size = images.shape[0]

        messages = messages.float().to(self.device)

        self.encoder_decoder.eval()
        self.discriminator.eval()

        if self.tb_logger is not None:
            encoder_final = dict(self.encoder_decoder.encoder.named_modules()).get('final_layer', None)
            if encoder_final is not None:
                self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = dict(self.encoder_decoder.decoder.named_modules()).get('linear', None)
            if decoder_final is not None:
                self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            discrim_final = dict(self.discriminator.named_modules()).get('linear', None)
            if discrim_final is not None:
                self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        with torch.no_grad():
            d_target_cover = torch.full((batch_size, 1), self.cover_label, dtype=torch.float32, device=self.device)
            d_target_encoded = torch.full((batch_size, 1), self.encoded_label, dtype=torch.float32, device=self.device)
            g_target_encoded = torch.full((batch_size, 1), self.cover_label, dtype=torch.float32, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_cover = self.bce_with_logits_loss(d_on_cover, d_target_cover)

            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            d_on_encoded = self.discriminator(encoded_images)
            d_loss_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_encoded)

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_encoded)

            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cover = self.vgg_loss(images)
                vgg_on_encoded = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cover, vgg_on_encoded)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().float().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().float().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_cover.item(),
            'discr_encod_bce': d_loss_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_string(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))