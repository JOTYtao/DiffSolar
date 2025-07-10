
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, omegaconf
from einops import rearrange
import numpy as np
from pathlib import Path
import argparse

from diffsolar.models.diffsolar import DiffSolar
from diffsolar.models.diffsolar_pipeline.pipeline import Pipeline
from diffsolar.models.diffsolar_pipeline.div_sampling import sampling

from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDIMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
from safetensors.torch import load_file
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_pretrained_pipeline(cfg, device):
    try:
        model = DiffSolar()
        state_dict = load_file(
            Path(cfg.TestCfg.ckpt_path) / "unet_ema" / "diffusion_pytorch_model.safetensors"
        )
        diffusion_unet_state_dict = {
            k.replace("diffusion_unet.", ""): v
            for k, v in state_dict.items()
            if k.startswith("diffusion_unet.")
        }
        model.diffusion_unet.load_state_dict(diffusion_unet_state_dict)
        model.vae.eval()
        requires_grad(model.vae, False)
        num_params = sum(p.numel() for p in model.diffusion_unet.parameters() if p.requires_grad)
        print('Number of trainable parameters in diffusion_unet:', num_params)
        return model
    except Exception as e:
        print(f"Error loading pipeline from {cfg.TestCfg.ckpt_path}: {e}")
        raise
def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--test_config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    return args.test_config
def inverse_rescale_data(scaled_data: torch.Tensor, min_val: float = 0.0, max_val: float = 1.2) -> torch.Tensor:
    return ((scaled_data + 1) / 2) * (max_val - min_val) + min_val
def main(cfg : DictConfig) -> None:
    accelerator = Accelerator()
    device = accelerator.device
    ckpt_path = cfg.TestCfg.ckpt_path
    r_save_path = cfg.TestCfg.test_results_path
    if not Path(r_save_path).exists():
        Path(r_save_path).mkdir(parents=True, exist_ok=True)
    #load stdiff model
    rediff = load_pretrained_pipeline(cfg, device=device)

    #Print the number of parameters
    num_params = sum(p.numel() for p in rediff.parameters() if p.requires_grad)
    print('Number of parameters is: ', num_params)

    #init scheduler
    if cfg.TestCfg.scheduler.name == 'DDIM':
        scheduler = DDIMScheduler.from_pretrained(ckpt_path, subfolder = 'scheduler')
    elif cfg.TestCfg.scheduler.name == 'DPMMS':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(ckpt_path, subfolder="scheduler", solver_order=3)
    elif cfg.TestCfg.scheduler.name == 'GR':
        scheduler = sampling.from_pretrained(ckpt_path, subfolder="scheduler")
    else:
        raise NotImplementedError("Scheduler is not supported")

    stdiff_pipeline = Pipeline(rediff, scheduler).to(device)

    if not accelerator.is_main_process:
        stdiff_pipeline.disable_pgbar()


    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _convert_="partial"
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    stdiff_pipeline, test_loader = accelerator.prepare(stdiff_pipeline, test_loader)

    if accelerator.is_main_process:
        test_config = {'cfg': cfg}
        torch.save(test_config, f=Path(r_save_path).joinpath('TestConfig.pt'))
    
    def get_resume_batch_idx(r_save_path):
        save_path = Path(r_save_path)
        saved_preds = sorted(list(save_path.glob('Preds_*')))
        saved_batches = sorted([int(str(p.name).split('_')[1].split('.')[0]) for p in saved_preds])
        try:
            return saved_batches[-1]
        except IndexError:
            return -1

    resume_batch_idx = get_resume_batch_idx(r_save_path)
    print('number of test batches: ', len(test_loader))
    print('resume batch index: ', resume_batch_idx)

    #Predict and save the predictions to disk for evaluation
    with torch.no_grad():
        progress_bar = tqdm(total=len(test_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Testing...")
        all_predictions = []
        all_targets = []
        all_inputs = []

        for idx, batch in enumerate(test_loader):
            # if idx > resume_batch_idx: #resume test
            if idx == 800:
                print(f"Processing batch {idx}")
                first_frame = batch["first_frame"].float()
                inputs = batch["his_cal"].float()
                target = batch["target"].float()
                #generator = torch.Generator(device=stdiff_pipeline.device).manual_seed(42)
                generator = torch.Generator(device=stdiff_pipeline.device)
                preds = []
                sp = []
                for i in range(cfg.TestCfg.random_predict.sample_num):
                    inputs_clone = inputs.clone()
                    first_frame_clone = first_frame.clone()
                    temp_pred, sp_pred = stdiff_pipeline(
                        first_frame_clone,
                        target,
                        inputs_clone,
                        generator=generator,
                        num_inference_steps=cfg.TestCfg.scheduler.sample_steps,
                        to_cpu=False,
                        fix_init_noise=cfg.TestCfg.random_predict.fix_init_noise
                    ) # B C T H W
                    preds.append(temp_pred)
                    sp.append(sp_pred)

                preds = torch.stack(preds, 0) #(sample_num, B, C, T, H, W)
                preds = preds.permute(1, 0, 2, 3, 4, 5).contiguous() #(B, sample_num, C, T, H, W)
                g_preds = accelerator.gather(preds)

                sp = torch.stack(sp, 0) #(sample_num, B, C, T, H, W)
                sp = sp.permute(1, 0, 2, 3, 4, 5).contiguous() #(B, sample_num, C, T, H, W)
                g_sp = accelerator.gather(sp)

                g_target = accelerator.gather(target)
                g_inputs = accelerator.gather(inputs)

                if accelerator.is_main_process:
                    save_path = Path(r_save_path)
                    g_target_denorm = inverse_rescale_data(g_target)
                    g_inputs_denorm = inverse_rescale_data(g_inputs)
                    np.save(save_path / f'sp_{idx}.npy', g_sp.detach().cpu().numpy())
                    np.save(save_path / f'predictions_{idx}.npy', g_preds.detach().cpu().numpy())
                    np.save(save_path / f'targets_{idx}.npy', g_target_denorm.detach().cpu().numpy())
                    all_predictions.append(g_preds.detach().cpu().numpy())
                    all_targets.append(g_target_denorm.detach().cpu().numpy())
                    all_inputs.append(g_inputs_denorm.detach().cpu().numpy())
                    progress_bar.update(1)
                    del g_preds
                    del g_target_denorm
                    del g_inputs_denorm
                    del g_sp
                    del g_target
                    del g_inputs
                break

        if accelerator.is_main_process:
            save_path = Path(r_save_path)
            all_predictions = np.concatenate(all_predictions, axis=0)
            np.save(save_path / 'all_predictions.npy', all_predictions)
            print(f"Predictions shape: {all_predictions.shape}")
            del all_predictions
            all_targets = np.concatenate(all_targets, axis=0)
            np.save(save_path / 'all_targets.npy', all_targets)
            print(f"Targets shape: {all_targets.shape}")
            del all_targets
            all_inputs = np.concatenate(all_inputs, axis=0)
            np.save(save_path / 'all_inputs.npy', all_inputs)
            print(f"Inputs shape: {all_inputs.shape}")
            del all_inputs
    print("Inference finished")
if __name__ == '__main__':
    config_path = Path(parse_args())
    initialize(version_base=None, config_path=str(config_path.parent))
    cfg = compose(config_name=str(config_path.name))
    main(cfg)