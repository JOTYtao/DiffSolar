import math
import argparse
import numpy as np
import torchvision
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from pathlib import Path
from typing import Any, Type
import torchmetrics
import os
from loss.contperceptual import LPIPSWithDiscriminator
REGISTERED_MODELS = {}
from diffusers import AutoencoderKL
REGISTERED_MODELS = {}

def register_model(cls: Type[pl.LightningModule]):
    global REGISTERED_MODELS
    name = cls.__name__
    assert name not in REGISTERED_MODELS, f"exists class: {REGISTERED_MODELS}"
    REGISTERED_MODELS[name] = cls
    return cls
class Autoencoder(pl.LightningModule):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 latent_channels=64,
                 layers_per_block=2,
                 # down_block_types=(
                 # "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
                 # up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
                 temporal_compression_ratio=4,
                 block_out_channels=(128, 256, 512, 512),
                 sample_size=64,
###############loss#######################################
                 disc_start=50001,
                 kl_weight=1e-6,
                 disc_weight=0.5,
                 learning_rate=4.5e-6,
                 save_dir: str = None,
                 visualize: bool = False,
                 ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.loss = LPIPSWithDiscriminator(
            disc_start=disc_start,
            kl_weight=kl_weight,
            disc_weight=disc_weight,
            perceptual_weight=0,
            disc_in_channels=1
        )
        self.visualize = visualize
        self.model = AutoencoderKL(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(128,256,512,512),
            sample_size=128,
            layers_per_block=2,
            
        )
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_step_outputs = []
        self.save_hyperparameters()

    @classmethod
    def from_config(cls, config):
        return Autoencoder(
            in_channels=config.get("in_channels"),
            out_channels=config.get("out_channels"),
            latent_channels=config.get("latent_channels"),

            disc_start=config.get("disc_start"),
            kl_weight=config.get("kl_weight"),
            disc_weight=config.get("disc_weight"),
            learning_rate=config.get("learning_rate"),
            save_dir=config.get("save_dir")
        )
    def get_last_layer(self):
        return self.model.decoder.conv_out.weight
    def forward(self, x, sample_posterior=True):

        latent = self.model.encode(x)
        posterior = latent.latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.model.decode(z)
        dec = dec.sample
        return dec, posterior

    def training_step(self, batch, batch_idx, optimizer_idx):

        inputs = batch["data"].float()
        B, C, T, H, W = inputs.shape
        inputs = inputs.reshape(B * T, C, H, W)
        
        reconstructions, posterior = self(inputs)
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs = batch["data"].float()
        B, C, T, H, W = inputs.shape
        inputs = inputs.reshape(B * T, C, H, W)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        inputs = batch["data"].float()
        B, C, T, H, W = inputs.shape
        inputs = inputs.reshape(B * T, C, H, W)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="test")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="test")
        self.log("test/rec_loss", log_dict_ae["test/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        inputs = inputs.reshape(B, C, T, H, W)
        reconstructions = reconstructions.reshape(B, C, T, H, W)
        self.test_mse(reconstructions, inputs)
        self.test_mae(reconstructions, inputs)
        output =  {"inputs": inputs.detach().cpu(), 
                   "posterior":  posterior.sample().detach().cpu(),
            "reconstructions": reconstructions.detach().cpu(),
            "aeloss": aeloss.detach().cpu()}
        self.test_step_outputs.append(output)
        return output
    @torch.no_grad()
    def inverse_rescale_data(self, scaled_data: torch.Tensor, min_val: float = 0.0, max_val: float = 1.2) -> torch.Tensor:
        return ((scaled_data + 1) / 2) * (max_val - min_val) + min_val
    @torch.no_grad()
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        all_inputs = torch.cat([output["inputs"] for output in outputs], dim=0)
        all_reconstructions = torch.cat([output["reconstructions"] for output in outputs], dim=0)
        all_posterior = torch.cat([output["posterior"] for output in outputs], dim=0)
        all_losses = torch.cat([output["aeloss"].unsqueeze(0) for output in outputs], dim=0)
        all_inputs_original = self.inverse_rescale_data(all_inputs, min_val=0.0, max_val=1.2)
        all_reconstructions_original = self.inverse_rescale_data(all_reconstructions, min_val=0.0, max_val=1.2)
        os.makedirs(self.save_dir, exist_ok=True)
        
        np.save(os.path.join(self.save_dir, "posterior.npy"), all_posterior.numpy())
        np.save(os.path.join(self.save_dir, "inputs.npy"), all_inputs_original.numpy())
        np.save(os.path.join(self.save_dir, "reconstructions.npy"), all_reconstructions_original.numpy())
        np.save(os.path.join(self.save_dir, "aelosses.npy"), all_losses.numpy())
        print(f"Test results saved to {self.save_dir}")
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.model.encoder.parameters())+
                                  list(self.model.decoder.parameters())+
                                  list(self.model.quant_conv.parameters())+
                                  list(self.model.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = batch["data"].float()
        x = x.to(self.device)
        B, C, T, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.model.decode(torch.randn_like(posterior.sample())).sample
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

