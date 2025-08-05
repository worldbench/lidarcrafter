import dataclasses
import datetime
import json
import os
import warnings
from pathlib import Path
import argparse
import einops
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
from rich import print
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sys
sys.path.append('../')

from lidargen.utils import inference
from lidargen.utils import render
from lidargen.utils import training
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets
warnings.filterwarnings("ignore", category=UserWarning)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.automatic_dynamic_shapes = False


def train(args):

    cfg = __all__[args.cfg]()
    cfg.resume = args.resume
    if args.batch_size is not None:
        cfg.training.batch_size_train = args.batch_size
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers

    torch.backends.cudnn.benchmark = True
    project_dir = Path(cfg.training.output_dir) / args.cfg /cfg.data.dataset / cfg.data.projection

    # =================================================================================
    # Initialize accelerator
    # =================================================================================

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=["tensorboard"],
        project_dir=project_dir,
        # dynamo_backend=cfg.training.dynamo_backend,
        # split_batches=True,
        step_scheduler_with_optimizer=True,
    )
    if accelerator.is_main_process:
        print(cfg)
        os.makedirs(project_dir, exist_ok=True)
        project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker("tensorboard")
        json.dump(
            dataclasses.asdict(cfg),
            open(Path(tracker.logging_dir) / "training_config.json", "w"),
            indent=4,
        )
    device = accelerator.device

    # =================================================================================
    # Setup models
    # =================================================================================
    channels = [
        1 if cfg.data.train_depth else 0,
        1 if cfg.data.train_reflectance else 0,
    ]

    if cfg.resume is not None:
        ddpm, model, lidar_utils, global_step, optimizer_dict, lr_scheduler_dict = inference.load_model_duffusion_training(cfg)

    else:
        ddpm, model, lidar_utils = inference.load_model_duffusion_training(cfg)

    ddpm.train()
    lidar_utils.train()
    ddpm.to(device)
    lidar_utils.to(device)        

    if accelerator.is_main_process:
        ddpm_ema = EMA(
            ddpm,
            beta=cfg.training.ema_decay,
            update_every=cfg.training.ema_update_every,
            update_after_step=cfg.training.lr_warmup_steps
            * cfg.training.gradient_accumulation_steps,
        )
        ddpm_ema.to(device)

    # =================================================================================
    # Setup optimizer & dataloader
    # =================================================================================

    optimizer = torch.optim.AdamW(
        ddpm.parameters(),
        lr=cfg.training.lr,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.training.adam_weight_decay,
        eps=cfg.training.adam_epsilon,
    )
    if cfg.resume is not None:
        optimizer.load_state_dict(optimizer_dict)

    cfg.data.split = 'all' # train val all
    dataset = all_datasets[cfg.data.dataset](cfg.data)

    if accelerator.is_main_process:
        print(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size_train,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=dataset.collate_fn if getattr(cfg.data, 'custom_collate_fn', False) else None
    )

    lr_scheduler = training.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps
        * cfg.training.gradient_accumulation_steps,
        num_training_steps=cfg.training.num_steps
        * cfg.training.gradient_accumulation_steps,
    )

    if cfg.resume is not None:
        lr_scheduler.load_state_dict(lr_scheduler_dict)

    ddpm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        ddpm, optimizer, dataloader, lr_scheduler
    )

    # =================================================================================
    # Utility
    # =================================================================================

    def preprocess(batch):
        x = []
        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(batch["depth"])]
        if cfg.data.train_reflectance:
            x += [batch["reflectance"]]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
        return x
    
    def preprocess_prev_cond(prev_cond):
        x = []
        reflectance = prev_cond[:,3,...]/255
        depth = prev_cond[:,-2,...]

        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(depth.unsqueeze(1))]
        if cfg.data.train_reflectance:
            x += [reflectance.unsqueeze(1)]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
        prev_labels = prev_cond[:,4,...].long()
        one_hot = F.one_hot(prev_labels, num_classes=len(list(cfg.data.class_names))+1).permute(0, 3, 1, 2)
        x = torch.cat((x, one_hot.float()), dim=1)
        return x

    def preprocess_autoregressive_cond(autoregressive_cond):
        x = []
        depth = autoregressive_cond[:, 0]
        reflectance = autoregressive_cond[:, 1]

        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(depth.unsqueeze(1))]
        if cfg.data.train_reflectance and args.cfg != 'nuscenes-auto-reg-v2':
            x += [reflectance.unsqueeze(1)]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
        return x

    def preprocess_condition_mask(batch):
        x = []
        condition_mask = batch['condition_mask'] # [B, 2, H, W]: semantic and depth
        # semantic
        curr_labels = condition_mask[:,0,...].long()
        one_hot = F.one_hot(curr_labels, num_classes=len(list(cfg.data.class_names))+1).permute(0, 3, 1, 2)
        x+= [one_hot.float()]
        # depth
        depth = lidar_utils.convert_depth(condition_mask[:,1,...].unsqueeze(1))
        x+= [depth]
        x = torch.cat(x, dim=1)
        return x

    def split_channels(image: torch.Tensor):
        depth, rflct = torch.split(image, channels, dim=1)
        return depth, rflct

    @torch.inference_mode()
    def log_images(image, tag: str = "name", global_step: int = 0):
        image = lidar_utils.denormalize(image)
        out = dict()
        depth, rflct = split_channels(image)
        if depth.numel() > 0:
            out[f"{tag}/depth"] = render.colorize(depth)
            metric = lidar_utils.revert_depth(depth)
            mask = (metric > lidar_utils.min_depth) & (metric < lidar_utils.max_depth)
            out[f"{tag}/depth/orig"] = render.colorize(
                metric / lidar_utils.max_depth
            )
            xyz = lidar_utils.to_xyz(metric) / lidar_utils.max_depth * mask
            normal = -render.estimate_surface_normal(xyz)
            normal = lidar_utils.denormalize(normal)
            bev = render.render_point_clouds(
                points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
                colors=einops.rearrange(normal, "B C H W -> B (H W) C"),
                t=torch.tensor([0, 0, 1.0]).to(xyz),
            )
            out[f"{tag}/bev"] = bev.mul(255).clamp(0, 255).byte()
        if rflct.numel() > 0:
            out[f"{tag}/reflectance"] = render.colorize(rflct, cm.plasma)
        if mask.numel() > 0:
            out[f"{tag}/mask"] = render.colorize(mask, cm.binary_r)
        tracker.log_images(out, step=global_step)

    # =================================================================================
    # Training loop
    # =================================================================================
    
    if cfg.resume is None:
        global_step = 0

    progress_bar = tqdm(
        range(global_step, cfg.training.num_steps),
        desc="training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    while global_step < cfg.training.num_steps:
        ddpm.train()
        for batch in dataloader:
            x_0 = preprocess(batch)
            batch['x_0'] = x_0
            if 'prev_cond' in batch:
                prev_cond = preprocess_prev_cond(batch['prev_cond'])
                batch['cond'] = prev_cond
            if 'condition_mask' in batch:
                condition_mask = preprocess_condition_mask(batch)
                batch['concat_cond'] = condition_mask

            if 'autoregressive_cond' in batch:
                autoregressive_cond = preprocess_autoregressive_cond(batch['autoregressive_cond'])
                batch['autoregressive_cond'] = autoregressive_cond

            with accelerator.accumulate(ddpm):
                loss = ddpm(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            log = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            if accelerator.is_main_process:
                ddpm_ema.update()
                log["ema/decay"] = ddpm_ema.get_current_decay()

                # if global_step == 1:
                #     log_images(x_0, "image", global_step)

                # if global_step % cfg.training.steps_save_image == 0:
                #     ddpm_ema.ema_model.eval()
                #     sample = ddpm_ema.ema_model.sample(
                #         batch_size=cfg.training.batch_size_eval,
                #         num_steps=cfg.diffusion.num_sampling_steps,
                #         rng=torch.Generator(device=device).manual_seed(0),
                #     )
                #     log_images(sample, "sample", global_step)

                if global_step % cfg.training.steps_save_model == 0:
                    save_dir = Path(tracker.logging_dir) / "models"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        {
                            "cfg": dataclasses.asdict(cfg),
                            "weights": ddpm_ema.online_model.state_dict(),
                            "ema_weights": ddpm_ema.ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "global_step": global_step,
                        },
                        save_dir / f"diffusion_{global_step:010d}.pth",
                    )

            accelerator.log(log, step=global_step)
            progress_bar.set_postfix(log)
            progress_bar.update(1)

            if global_step >= cfg.training.num_steps:
                break

    accelerator.end_training()

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-c",
        "--cfg",
        type=str,
        default="kitti-360",
        help="model config",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=None,
        help="batch size for training",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="worker number for dataset loading",
    )

    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="resume training from ckpt",
    )

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)
