# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from diffusers import utils
from diffusers import DiffusionPipeline, ImagePipelineOutput
import torchvision.transforms as transforms
from math import exp
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

class Pipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        
    @torch.no_grad()
    def __call__(
        self,
        first_frame,
        inputs,
        num_inference_steps: int = 1000,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        to_cpu=True,
        fix_init_noise=None
    ) -> Union[ImagePipelineOutput, Tuple]:
        x = self.model.encode(inputs)
        x = self.model.encode(inputs)
        sp_forecasting = torch.tile(inputs[:, :, -1:], [1, 1, 8, 1, 1])
        sp = self.model.encode(sp_forecasting)
        batch_size = inputs.shape[0]
        device = x.device
        res_shape = (batch_size, *sp.shape[1:])
        #set default value for fix_init_noise
        noisy_res = self.init_noise(res_shape, generator)
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            with torch.no_grad():
                timestep = torch.full((batch_size,), t, device=device).long()
                model_output = self.model.forward(
                    noisy_x=noisy_res,
                    timestep=timestep,
                    context=x
                )
            scheduler_output = self.scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=noisy_res,
            )
            noisy_res = scheduler_output.prev_sample
        pred = sp + noisy_res
        output = self.model.decode(pred)
        output_sp = self.model.decode(sp)
        
        output = self.inverse_rescale_data(scaled_data=output)
        output_sp = self.inverse_rescale_data(scaled_data=output_sp)


        if output_type == "numpy":
            output = output.cpu().numpy()
            output_sp = output_sp.cpu().numpy()
            return ImagePipelineOutput(images=output), ImagePipelineOutput(images=output_sp)
        else:
            return output, output_sp

    def inverse_rescale_data(self, scaled_data: torch.Tensor, min_val: float = 0.0, max_val: float = 1.2) -> torch.Tensor:
        return ((scaled_data + 1) / 2) * (max_val - min_val) + min_val
    def init_noise(self, image_shape, generator):
        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)
        
        return image
    
    def disable_pgbar(self):
        self.progress_bar = lambda x: x


def PSNR(x: Tensor, y: Tensor, data_range: Union[float, int] = 1.0, mean_flag: bool = True) -> Tensor:
    """
    Comput the average PSNR between two batch of images.
    x: input image, Tensor with shape (N, C, H, W)
    y: input image, Tensor with shape (N, C, H, W)
    data_range: the maximum pixel value range of input images, used to normalize
                pixel values to [0,1], default is 1.0
    """

    EPS = 1e-8
    x = x/float(data_range)
    y = y/float(data_range)

    mse = torch.mean((x-y)**2, dim = (1, 2, 3))
    score = -10*torch.log10(mse + EPS)
    if mean_flag:
        return torch.mean(score).item()
    else:
        return score

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
        self.__name__ = 'SSIM'
        
    def forward(self, img1: Tensor, img2: Tensor, mean_flag: bool = True) -> float:
        """
        img1: (N, C, H, W)
        img2: (N, C, H, W)
        Return:
            batch average ssim_index: float
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return self._ssim(img1, img2, window, self.window_size, channel, mean_flag)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window

    def _ssim(self, img1, img2, window, window_size, channel, mean_flag):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        if mean_flag:
            return ssim_map.mean()
        else:
            return torch.mean(ssim_map, dim=(1,2,3))