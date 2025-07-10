from torch import nn
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class ConvMlp(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=64, 
        out_features=None,
        kernel_size=3,
        act_layer=nn.GELU, 
        drop=0.0,
        num_groups=8 
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        padding = kernel_size // 2
    
        num_groups1 = min(num_groups, hidden_features//4)  
        num_groups2 = min(num_groups, out_features//4)
        
        self.conv1 = nn.Conv3d(
            in_features, hidden_features,
            kernel_size=kernel_size, 
            padding=padding,
            padding_mode='replicate'
        )
        self.norm1 = nn.GroupNorm(num_groups1, hidden_features)
        
        self.act = act_layer()

        self.conv2 = nn.Conv3d(
            hidden_features, out_features, 
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='replicate'
        )
        self.norm2 = nn.GroupNorm(num_groups2, out_features)
        
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        # [B,D,H,W,C] -> [B,C,D,H,W]
        x = x.permute(0,4,1,2,3)
        

        x = self.conv1(x)
        x = self.norm1(x) 
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.norm2(x) 
        x = self.drop(x)
        
        # [B,C,D,H,W] -> [B,D,H,W,C]
        x = x.permute(0,2,3,4,1)
        return x
class Mlp(nn.Module):
    def __init__(
        self, 
        in_features, hidden_features=None, out_features=None, 
        act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FFT3D(nn.Module):
    def __init__(
        self, hidden_size, num_blocks=8, sparsity_threshold=0.01,
        hard_thresholding_fraction=1, hidden_size_factor=1
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, D, H, W, C = x.shape

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x = x.reshape(B, D, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, D, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, D, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, D, H, W // 2 + 1, C)
        x = torch.fft.irfftn(x, s=(D, H, W), dim=(1,2,3), norm="ortho")
        x = x.type(dtype)

        return x + bias


class FourierUnit(nn.Module):
    def __init__(
            self,
            dim,
            kernel=3,
            mlp_ratio=4.,
            drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            mlp_out_features=None,
        ):
        super().__init__()
        self.norm_layer = norm_layer
        self.norm1 = norm_layer(dim)
        self.filter = FFT3D(dim, num_blocks, sparsity_threshold, 
            hard_thresholding_fraction) 
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convmlp = ConvMlp(
                in_features=dim,
                out_features=mlp_out_features,
                hidden_features=mlp_hidden_dim,
                kernel_size=kernel,
                act_layer=act_layer,
                drop=drop
            )
        self.double_skip = double_skip

    def forward(self, x):
        # AFNO natively uses a channels-last data format 
        x = x.permute(0,2,3,4,1)

        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.convmlp(x)
        x = x + residual
        x = x.permute(0,4,1,2,3)
        return x


class Inception(nn.Module):
    def __init__(self, C_in, C_out, incep_ker=[1,3,5]):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv3d(C_in,C_out, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(FourierUnit(dim=C_out, kernel=ker))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

class MSInception(nn.Module):
    def __init__(self, channel_in, channel_hid, channel_out, N_T, incep_ker=[1,3,5]):
        super(MSInception, self).__init__()
        
        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid, incep_ker)]
        for i in range(1, N_T):
            enc_layers.append(Inception(channel_hid, channel_hid, incep_ker))

        dec_layers = [Inception(channel_hid, channel_hid, incep_ker)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid, incep_ker))
        dec_layers.append(Inception(2*channel_hid, channel_out, incep_ker))
        
        self.enc = nn.ModuleList(enc_layers)
        self.dec = nn.ModuleList(dec_layers)
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channel_out, channel_out*2, kernel_size=3, stride=(1,2,2), padding=(1,1,1)),  # 16x16 -> 8x8
                nn.BatchNorm3d(channel_out*2),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(channel_out*2, channel_out*4, kernel_size=3, stride=(1,2,2), padding=(1,1,1)),  # 8x8 -> 4x4
                nn.BatchNorm3d(channel_out*4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(channel_out*4, channel_out*4, kernel_size=3, stride=(1,2,2), padding=(1,1,1)),  # 4x4 -> 2x2
                nn.BatchNorm3d(channel_out*4),
                nn.ReLU(inplace=True)
            )
        ])

    def forward(self, x):
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)
        z = self.dec[0](z)
    
        for i in range(1, self.N_T):

            z = torch.cat([z, skips[-i]], dim=1)
            z = self.dec[i](z)
        context = {}
        ms_feat = z 
        context[(16,16)] = ms_feat  
        for i, down in enumerate(self.downsample_layers):
            ms_feat = down(ms_feat)
            h = w = 16 // (2**(i+1))
            context[(h,h)] = ms_feat
        return context 
