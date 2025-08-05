import dataclasses
import datetime
import json
import os
import warnings
from pathlib import Path
import argparse
import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from rich import print
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sys
sys.path.append('/home/alan/AlanLiang/Projects/AlanLiang/LidarGen4D')
sys.path.append('/data/yyang/workspace/projects/LidarGen4D')
sys.path.append('/data1/liangao/AlanLiang/Projects/LidarGen4D')
from lidargen.utils import inference
from lidargen.utils import training
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets
warnings.filterwarnings("ignore", category=UserWarning)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.automatic_dynamic_shapes = False
from accelerate.utils import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

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

    # Single GPU training
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        # mixed_precision=cfg.training.mixed_precision,
        log_with=["tensorboard"],
        project_dir=project_dir,
        # dynamo_backend=cfg.training.dynamo_backend,
        # split_batches=True,
        step_scheduler_with_optimizer=True,
    )

    # Multi GPU training
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
    #     # mixed_precision='fp16',
    #     log_with=["tensorboard"],
    #     project_dir=project_dir,
    #     step_scheduler_with_optimizer=True,
    #     kwargs_handlers=[ddp_kwargs]
    #     )

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
    # Setup dataset
    # =================================================================================

    cfg.data.split = 'train'
    dataset = all_datasets[cfg.data.dataset](cfg.data)
    if accelerator.is_main_process:
        print(dataset)
    cfg.condition_model.params['vocab'] = dataset.scene_graph_assigner.vocab

    # =================================================================================
    # Setup models
    # =================================================================================

    if cfg.resume is not None:
        ddpm, model, global_step, optimizer_dict, lr_scheduler = inference.load_model_layout_duffusion_training(cfg)

    else:
        ddpm, model = inference.load_model_layout_duffusion_training(cfg)

    ddpm.train()
    ddpm.to(device)

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

    ddpm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        ddpm, optimizer, dataloader, lr_scheduler
    )

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
