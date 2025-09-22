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
        output_type: Optional[str] = "numpy",
        to_cpu=True,
        fix_init_noise=None
    ) -> Union[ImagePipelineOutput, Tuple]:
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



        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        if mean_flag:
            return ssim_map.mean()
        else:
            return torch.mean(ssim_map, dim=(1,2,3))
