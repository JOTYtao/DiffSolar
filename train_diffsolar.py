import inspect
import logging
import math
import os
from pathlib import Path
import numpy as np
import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm
import argparse
from pytorch_lightning import (
    LightningDataModule,
)
from accelerate.utils import DistributedDataParallelKwargs
import diffusers
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available
import wandb
from hydra.core.global_hydra import GlobalHydra
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from diffsolar.models.diffsolar_pipeline import Pipeline
from diffsolar.models.diffsolar import DiffSolar
import warnings
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")
check_min_version("0.15.0.dev0")
logger = get_logger(__name__, log_level="INFO")
def inverse_rescale_data(scaled_data: torch.Tensor, min_val: float = 0.0, max_val: float = 1.2) -> torch.Tensor:
    return ((scaled_data + 1) / 2) * (max_val - min_val) + min_val
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
def initialize_wandb(cfg):
    wandb.init(
        project=cfg.Env.wandb.project,
        name=cfg.Env.wandb.name,
        entity=cfg.Env.wandb.entity,
        tags=cfg.Env.wandb.tags,
        notes=cfg.Env.wandb.notes,
        group=cfg.Env.wandb.group,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_code=cfg.Env.wandb.save_code,
        mode=cfg.Env.wandb.mode,
        dir=cfg.Env.output_dir  
    )
def parse_args():
    parser = argparse.ArgumentParser(description="Solar Flow Training Script")
    parser.add_argument(
        '--train_config',
        type=str,
        default="diffsolar/configs/SSI_train_config.yaml",
        help='Path to configuration file (default: configs/SSI_train_config.yaml)'
    )
    args = parser.parse_args()
    return args.train_config

@torch.no_grad()
def validate(model, val_dataloader, noise_scheduler, accelerator, epoch, global_step, cfg):
    model.eval()
    val_losses = []
    
    for batch in tqdm(val_dataloader, desc="Validating", disable=not accelerator.is_local_main_process):
        first_frame = batch["first_frame"].float()
        inputs = batch["his"].float()
        target = batch["target"].float()
                
        x = model.module.encode(inputs)
        sp_past = torch.tile(first_frame[:, :, -1:], [1, 1, 7, 1, 1])
        sp_past_latent = model.module.encode(sp_past)
        res_past = x - sp_past_latent
                
        y = model.module.encode(target)
        sp_forecasting = torch.tile(inputs[:, :, -1:], [1, 1, 8, 1, 1])
        sp = model.module.encode(sp_forecasting)
        res = y - sp
        noise = torch.randn(res.shape).to(res.device)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (res.shape[0],), device=res.device).long()
        noisy_res= noise_scheduler.add_noise(res, noise, timesteps)
        model_output = model.forward(noisy_x=noisy_res, timestep=timesteps, context=x)
        
        # Calculate loss
        if cfg.STDiff.Diffusion.prediction_type == "epsilon":
            val_loss = F.l1_loss(model_output, noise)
        elif cfg.STDiff.Diffusion.prediction_type == "sample":
            alpha_t = _extract_into_tensor(
                noise_scheduler.alphas_cumprod, timesteps, (y.shape[0], 1, 1, 1)
            )
            snr_weights = alpha_t / (1 - alpha_t)
            val_loss = snr_weights * F.l1_loss(model_output, y, reduction="none")
            val_loss = val_loss.mean()
            
        val_losses.append(val_loss.item())
    
    # Calculate average validation loss
    avg_val_loss = sum(val_losses) / len(val_losses)
    
    # Log validation results
    logs = {
        "val_loss": avg_val_loss,
        "epoch": epoch,
    }
    
    accelerator.log(logs, step=global_step)
    
    # Generate validation samples
    if accelerator.is_main_process and (epoch % cfg.Training.save_images_epochs == 0 or epoch == cfg.Training.epochs - 1):
        unet = accelerator.unwrap_model(model)
        if hasattr(unet, 'module'):
            unet = unet.module
            
        pipeline = Pipeline(
            model=unet,
            scheduler=noise_scheduler,
        )
        
        
        
        # Generate samples from validation data
        batch = next(iter(val_dataloader))
        first_frame = batch["first_frame"].float()
        inputs = batch["his_cal"].float()
        target = batch["target"].float()
        
        with torch.no_grad():
            images = pipeline(
                first_frame,
                inputs,
                generator=torch.Generator(device=pipeline.device).manual_seed(42),
                num_inference_steps=cfg.DiffSolar.Diffusion.ddim_num_inference_steps,
                output_type="numpy"
            )
        
        # Process and log validation images
        inputs = inverse_rescale_data(inputs)
        target = inverse_rescale_data(target)
        images = images.images
        
        if cfg.Env.logger == "wandb":
            def create_time_grid(data):
                B, C, T, H, W = data.shape
                grid_size = int(np.ceil(np.sqrt(T)))
                grid = np.zeros((H, W * T, C))
                for t in range(T):
                    w_start, w_end = t*W, (t+1)*W
                    current_slice = np.transpose(data[0, :, t], (1, 2, 0))  # [C,H,W] -> [H,W,C]
                    grid[:, w_start:w_end] = current_slice
                return grid if C > 1 else grid.squeeze(-1)
            
            tracker = accelerator.get_tracker("wandb")
            grid_pred = create_time_grid(images)
            grid_target = create_time_grid(target.cpu().numpy())
            
            tracker.log({
                "val_predictions": wandb.Image(grid_pred, caption="Validation predictions"),
                "val_targets": wandb.Image(grid_target, caption="Validation targets")
            }, step=global_step)
    
    return avg_val_loss
def main(cfg : DictConfig) -> None:
    logging_dir = os.path.join(cfg.Env.output_dir, 'logs')

    accelerator_project_config = ProjectConfiguration(total_limit=cfg.Training.epochs // cfg.Training.save_model_epochs)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.Training.gradient_accumulation_steps,
        mixed_precision=cfg.Training.mixed_precision,
        log_with=cfg.Env.logger,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False
            )
        ]
    )

    if cfg.Env.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if cfg.Training.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if cfg.Training.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), DiffSolar)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = DiffSolar.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.Env.output_dir is not None:
            os.makedirs(cfg.Env.output_dir, exist_ok=True)

    # Initialize the model
    model = DiffSolar()
    num_p_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num params of DiffSolar: {num_p_model/1e6} M')

    if cfg.Env.DiffSolar_init_ckpt is not None:
        model = DiffSolar.from_pretrained(cfg.Env.DiffSolar_init_ckpt, subfolder='unet')
        print('Init from a checkpoint')

    # Create EMA for the model.
    if cfg.Training.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=cfg.Training.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=cfg.Training.ema_inv_gamma,
            power=cfg.Training.ema_power,
            model_cls=DiffSolar,
            model_config=model.config,
        )

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDIMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.DiffSolar.Diffusion.ddim_num_steps,
            beta_schedule=cfg.DiffSolar.Diffusion.ddim_beta_schedule,
            prediction_type=cfg.DiffSolar.Diffusion.prediction_type,
            clip_sample=False, 
            set_alpha_to_one=False 
        )
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.DiffSolar.Diffusion.ddim_num_steps,
            beta_schedule=cfg.DiffSolar.Diffusion.ddpm_beta_schedule
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.Training.learning_rate,
        betas=cfg.Training.adam_betas,
        weight_decay=cfg.Training.adam_weight_decay,
        eps=cfg.Training.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # Preprocessing the datasets and DataLoaders creation.
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _convert_="partial"
    )
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        cfg.Training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.Training.lr_warmup_steps * cfg.Training.gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * cfg.Training.epochs,
        num_cycles=cfg.Training.num_cycles,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if cfg.Training.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = cfg.datamodule.batch_size * accelerator.num_processes * cfg.Training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.Training.gradient_accumulation_steps)
    max_train_steps = cfg.Training.epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {cfg.Training.epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.datamodule.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.Training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.Env.resume_ckpt is None:
        accelerator.print(
            f"Starting a new training run."
        )
        cfg.Env.resume_ckpt = None
    else:
        accelerator.print(f"Resuming from checkpoint {cfg.Env.resume_ckpt}")
        accelerator.load_state(os.path.join(cfg.Env.output_dir, cfg.Env.resume_ckpt))
        global_step = int(cfg.Env.resume_ckpt.split("-")[1])

        resume_global_step = global_step * cfg.Training.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * cfg.Training.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, cfg.Training.epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                first_frame = batch["first_frame"].float()
                inputs = batch["his"].float()
                target = batch["target"].float()
                
                x = model.module.encode(inputs)
                sp_past = torch.tile(first_frame[:, :, -1:], [1, 1, 7, 1, 1])
                sp_past_latent = model.module.encode(sp_past)
                # res_past = x - sp_past_latent
                
                y = model.module.encode(target)
                sp_forecasting = torch.tile(inputs[:, :, -1:], [1, 1, 8, 1, 1])
                sp = model.module.encode(sp_forecasting)
                res = y - sp
                
                
                # Skip steps until we reach the resumed step
                if cfg.Env.resume_ckpt and epoch == first_epoch and step < resume_step:
                    if step % cfg.Training.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                noise = torch.randn(res.shape).to(res.device)
                bsz = res.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=res.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_res= noise_scheduler.add_noise(res, noise, timesteps)
                # Predict the noise residual
                model_output = model.forward(noisy_x=noisy_res, timestep=timesteps, context=x)
                if cfg.DiffSolar.Diffusion.prediction_type == "epsilon":
                    loss = F.l1_loss(model_output, noise)  # this could have different weights!
                elif cfg.DiffSolar.Diffusion.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (res.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.l1_loss(
                        model_output, res, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {cfg.DiffSolar.Diffusion.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.Training.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % cfg.Training.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(cfg.Env.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if cfg.Training.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()
        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % cfg.Training.save_images_epochs == 0 or epoch == cfg.Training.epochs - 1:
                unet = accelerator.unwrap_model(model)
                if hasattr(unet, 'module'):
                    unet = unet.module
                if cfg.Training.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = Pipeline(
                    model=unet,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(42)
                # run pipeline in inference (sample random noise and denoise)

                batch = next(iter(train_dataloader))
                first_frame = batch["first_frame"].float()
                inputs = batch["his"].float()
                target = batch["target"].float()
                images, osp = pipeline(
                    first_frame,
                    inputs,
                    generator=generator,
                    num_inference_steps=cfg.DiffSolar.Diffusion.ddim_num_inference_steps,
                    output_type="numpy"
                )
                                
                inputs = inverse_rescale_data(inputs)
                target = inverse_rescale_data(target)
                images = images.images
                osp = osp.images
                batch_size, channels, time_steps, height, width = images.shape
                def create_grid(data):
                    batch_size, channels, time_steps, height, width = data.shape
                    grid_size = int(np.ceil(np.sqrt(time_steps)))
                    grid_height = grid_size * height
                    grid_width = grid_size * width
                    grid_images = np.zeros((batch_size, channels, grid_height, grid_width))
                    for t in range(time_steps):
                        i, j = t // grid_size, t % grid_size
                        grid_images[:, :, i * height:(i + 1) * height, j * width:(j + 1) * width] = data[:, :, t, :, :]
                    return grid_images
                grid_inputs = create_grid(inputs.cpu().numpy())
                grid_images = create_grid(images)
                grid_osp = create_grid(osp)
                grid_targets = create_grid(target.cpu().numpy())
                if cfg.Training.use_ema:
                    ema_model.restore(unet.parameters())
                if cfg.Env.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("inputs", grid_inputs, epoch)
                    tracker.add_images("predictions", grid_images, epoch)
                    tracker.add_images("sp_predictions", grid_osp, epoch)
                    tracker.add_images("targets", grid_targets, epoch)
                    for t in range(time_steps):
                        tracker.add_images(f"time_{t}/prediction", images[:, :, t], epoch)
                        tracker.add_images(f"time_{t}/osp", osp[:, :, t], epoch)
                        tracker.add_images(f"time_{t}/target", target[:, :, t].cpu().numpy(), epoch)
                     
                elif cfg.Env.logger == "wandb":
                    initialize_wandb(cfg)
                    def create_time_grid(data):

                        B, C, T, H, W = data.shape
                        grid_size = int(np.ceil(np.sqrt(T)))
                        grid = np.zeros((grid_size * H, grid_size * W, C))
                        for t in range(T):
                            i, j = t // grid_size, t % grid_size
                            h_start, h_end = i*H, (i+1)*H
                            w_start, w_end = j*W, (j+1)*W
                            current_slice = np.transpose(data[0, :, t], (1, 2, 0))  # [C,H,W] -> [H,W,C]
                            grid[h_start:h_end, w_start:w_end] = current_slice
                        return grid if C > 1 else grid.squeeze(-1)
                    tracker = accelerator.get_tracker("wandb")
                    grid_pred = create_time_grid(images)
                    grid_osp = create_time_grid(osp)
                    grid_target = create_time_grid(target.cpu().numpy())
                    tracker.log({
                        "all_predictions": wandb.Image(grid_pred, caption="All time steps predictions"),
                        "all_osp": wandb.Image(grid_osp, caption="All time steps OSP"),
                        "all_targets": wandb.Image(grid_target, caption="All time steps targets")
                    }, step=global_step)
                    tracker.log({"epoch": epoch}, step=global_step)

            if epoch % cfg.Training.save_model_epochs == 0 or epoch == cfg.Training.epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if cfg.Training.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = Pipeline(
                    model=unet,
                    scheduler=noise_scheduler
                )

                pipeline.save_pretrained(cfg.Env.output_dir)

                if cfg.Training.use_ema:
                    ema_model.restore(unet.parameters())

    accelerator.end_training()


if __name__ == '__main__':
    try:
        config_path = Path(parse_args())
        config_dir = config_path.parent
        config_name = config_path.stem
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize(version_base=None, config_path=str(config_dir))
        cfg = compose(config_name=config_name)
        main(cfg)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise