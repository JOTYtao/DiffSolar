import torch
import torchvision.transforms as transforms
from diffusers.configuration_utils import register_to_config
from diffusers import ConfigMixin, ModelMixin
from omegaconf import OmegaConf
from diffusers import AutoencoderKL
from diffsolar.models.denoiser.unets.denoiser import Denoiser
import warnings
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
class DiffSolar(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,):
        super().__init__()
        self.diffusion_unet = Denoiser(
            sample_size=16,
            in_channels=32,
            out_channels=32,
            down_block_types= (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
            ),
            up_block_types=(
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            ),
            block_out_channels=(128, 256, 512, 512),
            N_T=3,
            cross_attention_dim=512,
            channel_hid=256
        )
        self.vae = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            latent_channels=32,
            layers_per_block=2,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=[128, 256, 512, 512],
            sample_size=128,
        )
        self.vae_path = 'vae.ckpt'
        if self.vae_path is not None:
            state_dict = torch.load(self.vae_path, map_location=torch.device("cpu"))
            new_state_dict = {k.replace("model.", ""): v for k, v in state_dict['state_dict'].items()}
            self.vae.load_state_dict(new_state_dict, strict=False)
        else:
            warnings.warn(f"Pretrained weights for `AutoencoderKL` not set. Run for sanity check only.")
        self.vae.eval()
        requires_grad(self.vae, False)

    @torch.no_grad()
    def encode(self, x):
        if len(x.shape) == 5:  # [B, C, T, H, W]
            B, C, T, H, W = x.shape
            latent_z = []
            for t in range(T):
                current_images = x[:, :, t, :, :]
                with torch.no_grad():
                    current_z = self.vae.encode(current_images).latent_dist.sample()  # Scaling factor computed as 1/std of encoder outputs
                    latent_z.append(current_z)
            return torch.stack(latent_z, dim=2)  # [B, C, T, latent_H, latent_W]
        elif len(x.shape) == 4:
            with torch.no_grad():
                latent_z = self.vae.encode(x).latent_dist.sample()
            return latent_z
        else:
            raise ValueError(f"Invalid input shape: {x.shape}. Expected [B, C, T, H, W] or [B, C, H, W].")

    @torch.no_grad()
    def decode(self, latent_z):
        if len(latent_z.shape) == 5:  # [B, C, T, latent_H, latent_W]
            B, C, T, latent_H, latent_W = latent_z.shape
            reconstructed_images = []
            for t in range(T):
                current_latent = latent_z[:, :, t, :, :]
                with torch.no_grad():
                    current_image = self.vae.decode(current_latent).sample
                    reconstructed_images.append(current_image)
            return torch.stack(reconstructed_images, dim=2)
        elif len(latent_z.shape) == 4:
            with torch.no_grad():
                reconstructed_image = self.vae.decode(latent_z).sample
            return reconstructed_image
        else:
            raise ValueError(
                f"Invalid latent representation shape: {latent_z.shape}. Expected [B, C, T, latent_H, latent_W] or [B, C, latent_H, latent_W].")

    def forward(self, noisy_x, timestep, context):
        out = self.diffusion_unet(sample=noisy_x, timestep=timestep, his_seq=context)
        return out
