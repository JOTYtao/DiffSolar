import torch
import torch.nn as nn
from typing import Union, Tuple, Dict
from diffsolar.models.denoiser.unets.MSInception import MSInception
from diffsolar.models.denoiser.unets.unet_spatio_temporal import (
    UNetSpatioTemporalRopeConditionModel,
    UNetSpatioTemporalRopeConditionOutput
)

class Denoiser(nn.Module):
    def __init__(
        self,
        sample_size: int = 16,
        in_channels: int = 32,
        out_channels: int = 32,
        num_frames: int = 8,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        num_attention_heads: Union[int, Tuple[int]] = (8, 8, 8, 8),
        N_T: int = 3,
        channel_hid: int = 256,
    ):
        super().__init__()
        
        # Context Network
        self.contextnet = MSInception(
            channel_in=in_channels,
            channel_hid=channel_hid,
            channel_out=block_out_channels[0],
            N_T=N_T
        )
        
        # UNet Model
        self.unetmodel = UNetSpatioTemporalRopeConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_frames=num_frames,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim,
        )
        
    def forward(
        self,
        sample: torch.FloatTensor,
        his_seq: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
        position_ids=None,
    ) -> Union[UNetSpatioTemporalRopeConditionOutput, Tuple]:
        """
        Args:
            sample (torch.FloatTensor): Input tensor with shape (batch, num_frames, channel, height, width)
            timestep (Union[torch.Tensor, float, int]): Current timestep in the diffusion process
            return_dict (bool, optional): Whether to return a dictionary. Defaults to True.
            position_ids (torch.Tensor, optional): Position IDs for the transformer. Defaults to None.
            
        Returns:
            Union[UNetSpatioTemporalRopeConditionOutput, Tuple]: Denoised output
        """
        # Get multi-scale context features
        sample = sample.permute(0, 2, 1, 3, 4)
        context_features = self.contextnet(his_seq)
        
        # Forward pass through UNet with context features
        output = self.unetmodel(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=context_features,
            return_dict=return_dict,
            position_ids=position_ids,
        ).sample
        output = output.permute(0, 2, 1, 3, 4)
        
        return output