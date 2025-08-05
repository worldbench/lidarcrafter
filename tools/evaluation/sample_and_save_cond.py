import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from rich import print
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import sys
sys.path.append("../")
from lidargen.utils import inference
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets
warnings.filterwarnings("ignore", category=UserWarning)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.automatic_dynamic_shapes = False


def sample(args):
    cfg = __all__[args.cfg]()
    cfg.resume = args.ckpt
    if args.batch_size is not None:
        cfg.training.batch_size_train = args.batch_size
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        # dynamo_backend=cfg.training.dynamo_backend,
        split_batches=True,
        step_scheduler_with_optimizer=True,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(cfg)


    ddpm, model, lidar_utils, _, _, _ = inference.load_model_duffusion_training(cfg)
    ddpm.eval()
    lidar_utils.eval()
    ddpm.to(device)
    lidar_utils.to(device)

    cfg.data.split = 'all'
    cfg.data.num_sample = args.num_sample
    dataset = all_datasets[cfg.data.dataset](cfg.data)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size_train,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        drop_last=False,
        collate_fn=dataset.collate_fn if getattr(cfg.data, 'custom_collate_fn', False) else None
    )

    lidar_utils, dataloader = accelerator.prepare(lidar_utils, dataloader)
    sample_fn = torch.compile(ddpm.sample)

    save_dir = Path(args.output_dir)
    with accelerator.main_process_first():
        save_dir.mkdir(parents=True, exist_ok=True)

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

    def postprocess(sample):
        sample = lidar_utils.denormalize(sample)
        depth, rflct = sample[:, [0]], sample[:, [1]]
        depth = lidar_utils.revert_depth(depth)
        xyz = lidar_utils.to_xyz(depth)
        return torch.cat([depth, xyz, rflct], dim=1)

    global_step = 0
    for batch in tqdm(
        dataloader,
        desc="saving...",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    ):
        global_seed = global_step * args.batch_size
        seeds = [global_seed + i for i in range(args.batch_size)]

        x_0 = preprocess(batch)
        batch['x_0'] = x_0
        if 'prev_cond' in batch:
            prev_cond = preprocess_prev_cond(batch['prev_cond'])
            batch['cond'] = prev_cond
        if 'condition_mask' in batch:
            condition_mask = preprocess_condition_mask(batch)
            batch['concat_cond'] = condition_mask

        with torch.cuda.amp.autocast(cfg.training.mixed_precision is not None):
            samples = sample_fn(
                batch_dict=batch,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                mode=args.mode,
                # rng=inference.setup_rng(seeds, device=device),
                progress=True,
            ).clamp(-1, 1)

        samples = postprocess(samples)

        for i in range(len(samples)):
            sample = samples[i]
            torch.save(sample.clone(), save_dir / f"samples_{seeds[i]:07d}_{batch['token'][i]}.pth")
        global_step += 1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--output_dir", default='../generated_results/our', type=str)
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
        "--num_sample",
        type=int,
        default=10000,
        help="Sampling number for evaluation (default: 10000)",
    )
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--mode", choices=["ddpm", "ddim"], default="ddpm")
    args = parser.parse_args()

    sample(args)
