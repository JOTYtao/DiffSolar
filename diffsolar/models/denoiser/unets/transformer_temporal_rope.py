
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffsolar.models.denoiser.unets.resnet import AlphaBlender
from diffsolar.models.denoiser.unets.attention import (
    BasicTransformerBlock,
    TemporalRopeBasicTransformerBlock,
)
def rope(pos: torch.Tensor, dim: int, theta=10000.0) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    # (B, N, d/2)
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    # (B, 1, N, d/2, 2, 2)
    out = stacked_out.view(batch_size, 1, -1, dim // 2, 2, 2)
    return out.float()


@dataclass
class TransformerTemporalModelOutput(BaseOutput):
    """
    The output of [`TransformerTemporalModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size x num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input.
    """

    sample: torch.FloatTensor


class TransformerSpatioTemporalModel(nn.Module):
   
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(
            num_groups=8, num_channels=in_channels, eps=1e-6
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)
        
        self.encoder_norm = torch.nn.GroupNorm(
            num_groups=8, num_channels=in_channels, eps=1e-6
        )
        self.encoder_proj = nn.Linear(in_channels, cross_attention_dim)


        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for d in range(num_layers)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalRopeBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        position_ids: Optional[torch.Tensor] = None,
    ):
       
        # 1. Input
        batch_frames, channels, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        # (B*F, 1, C)
        # time_context = encoder_hidden_states
        # (B, 1, C)
        # time_context_first_timestep = time_context[None, :].reshape(
        #     batch_size, num_frames, -1, time_context.shape[-1]
        # )[:, 0]

        # (B*N, 1, C)
        # time_context = time_context_first_timestep.repeat_interleave(
        #     height * width, dim=0
        # )

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch_frames, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)
        
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.encoder_norm(encoder_hidden_states)

            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 3, 1).reshape(
                batch_frames, height * width, channels
            )
            encoder_hidden_states = self.encoder_proj(encoder_hidden_states)

            time_context = encoder_hidden_states.reshape(
                batch_size, num_frames, height * width, -1
            )[:, 0]  # [B, H*W, C]
            time_context = time_context.repeat_interleave(num_frames, dim=0)  # [B*F, H*W, C]
        if position_ids is None:
            # (B, F)
            frame_rotary_emb = torch.arange(num_frames, device=hidden_states.device)
            frame_rotary_emb = frame_rotary_emb[None, :].repeat(batch_size, 1)
        else:
            frame_rotary_emb = position_ids

        # (B, 1, F, d/2, 2, 2)
        frame_rotary_emb = rope(frame_rotary_emb, self.attention_head_dim)
        # (B*N, 1, F, d/2, 2, 2)
        frame_rotary_emb = frame_rotary_emb.repeat_interleave(height * width, dim=0)
        # 2. Blocks
        for block, temporal_block in zip(
            self.transformer_blocks, self.temporal_transformer_blocks
        ):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states_mix = temporal_block(
                hidden_states,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
                frame_rotary_emb=frame_rotary_emb,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )
        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch_frames, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        output = hidden_states + residual
        if not return_dict:
            return (output,)
        
        
        return TransformerTemporalModelOutput(sample=output)
