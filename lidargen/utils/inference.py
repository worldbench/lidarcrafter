from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lidargen.utils.lidar import LiDARUtility, get_linear_ray_angles
from lidargen.models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
    CondContinuousTimeGaussianDiffusion,
    CondContinuousLayoutGaussianDiffusion,
    CondContinuousLayoutGaussianDiffusion1D
)
from lidargen.dataset import __all__ as all_datasets
from lidargen.models.refinenet import LiDARGenRefineNet
from lidargen.utils.lidar import LiDARUtility
from lidargen.models.unets import EfficientUNet
from .configs import __all__
from lidargen.models.unets import __all__ as __all_unets__
from lidargen.models.dits import __all__ as __all_dits__
from lidargen.models.flows import __all__ as __all_flows__

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_model(
    cfg: str,
    ckpt: str | Path | dict,
    device: torch.device | str = "cpu",
    ema: bool = True,
    show_info: bool = True,
    compile: bool = False,
):# -> tuple[GaussianDiffusion, LiDARUtility, Config]:
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location="cpu")
    cfg = __all__[cfg](**ckpt["cfg"])

    in_channels = [0, 0]
    if cfg.data.train_depth:
        in_channels[0] = 1
    if cfg.data.train_reflectance:
        in_channels[1] = 1
    in_channels = sum(in_channels)

    if 'dit' in cfg.model.architecture:
        model = __all_dits__[cfg.model.architecture](in_channels=in_channels,
                                                resolution=cfg.data.resolution,
                                                **cfg.model.params)
    else:
        model = __all_unets__[cfg.model.architecture](in_channels=in_channels,
                                                resolution=cfg.data.resolution,
                                                **cfg.model.params)

    if cfg.diffusion.timestep_type == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            model=model,
            loss_type=cfg.diffusion.loss_type,
            num_training_steps=cfg.diffusion.num_training_steps,
            prediction_type=cfg.diffusion.prediction_type,
            noise_schedule=cfg.diffusion.noise_schedule,
        )
    elif cfg.diffusion.timestep_type == "continuous":
        ddpm = ContinuousTimeGaussianDiffusion(
            model=model,
            condition_model=nn.Identity(),
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,
            noise_schedule=cfg.diffusion.noise_schedule,
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion.timestep_type}")

    state_dict = ckpt["ema_weights"] if ema else ckpt["weights"]
    ddpm.load_state_dict(state_dict)
    ddpm.eval()
    ddpm.to(device)

    if compile:
        ddpm.model = torch.compile(ddpm.model)

    lidar_utils = LiDARUtility(
        resolution=cfg.data.resolution,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=ddpm.model.coords,
    )
    lidar_utils.eval()
    lidar_utils.to(device)

    if show_info:
        print(
            *[
                f"resolution: {model.resolution}",
                f"model: {model.__class__.__name__}",
                f"ddpm: {ddpm.__class__.__name__}",
                f'#steps:  {ckpt["global_step"]:,}',
                f"#params: {count_parameters(ddpm):,}",
            ],
            sep="\n",
        )

    return ddpm, lidar_utils, cfg

def setup_model_flow(
    cfg: str,
    ckpt: str | Path | dict,
    device: torch.device | str = "cpu",
    ema: bool = True,
    show_info: bool = True,
    compile: bool = False,
    ):# -> tuple[GaussianDiffusion, LiDARUtility, Config]:    

    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location="cpu")
    cfg = __all__[cfg](**ckpt["cfg"])
    flow, model, lidar_utils = load_model_flow_training(cfg)

    state_dict = ckpt["ema_weights"] if ema else ckpt["weights"]
    flow.load_state_dict(state_dict)
    flow.eval()
    flow.to(device)
    lidar_utils.eval()
    lidar_utils.to(device)

    if show_info:
        print(
            *[
                f"resolution: {model.resolution}",
                f"model: {model.__class__.__name__}",
                f"ddpm: {flow.__class__.__name__}",
                f'#steps:  {ckpt["global_step"]:,}',
                f"#params: {count_parameters(flow):,}",
            ],
            sep="\n",
        )
    return flow, lidar_utils, cfg

def setup_model_dataset_cond(
    cfg: str,
    ckpt: str | Path | dict,
    device: torch.device | str = "cpu",
    ema: bool = True,
    show_info: bool = True,
    compile: bool = False,
    split: str = 'train'
):# -> tuple[GaussianDiffusion, LiDARUtility, Config]:
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location="cpu")
    cfg = __all__[cfg]()

    in_channels = [0, 0]
    if cfg.data.train_depth:
        in_channels[0] = 1
    if cfg.data.train_reflectance:
        in_channels[1] = 1
    in_channels = sum(in_channels)

    if hasattr(cfg, 'condition_model') and getattr(cfg.diffusion, 'cond_mode', None) == 'concat':
        in_channels += cfg.condition_model.params['out_channels']

    if 'dit' in cfg.model.architecture:
        model = __all_dits__[cfg.model.architecture](in_channels=in_channels,
                                                resolution=cfg.data.resolution,
                                                **cfg.model.params)
    else:
        model = __all_unets__[cfg.model.architecture](in_channels=in_channels,
                                                resolution=cfg.data.resolution,
                                                **cfg.model.params)
    
    if "spherical" in cfg.data.projection:
        model.coords = get_linear_ray_angles(H=cfg.data.resolution[0], W=cfg.data.resolution[1], fov_up=cfg.data.fov_up, fov_down=cfg.data.fov_down)
    elif "unfolding" in cfg.data.projection:
        model.coords = F.interpolate(
            torch.load(f"data/{cfg.data.dataset}/unfolding_angles.pth"),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
    else:
        raise ValueError(f"Unknown: {cfg.data.projection}")

    if hasattr(cfg, 'condition_model'):
        condition_model = __all_unets__[cfg.condition_model.architecture](**cfg.condition_model.params)
    else:
        condition_model = nn.Identity()

    if cfg.diffusion.timestep_type == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            model=model,
            loss_type=cfg.diffusion.loss_type,
            num_training_steps=cfg.diffusion.num_training_steps,
            prediction_type=cfg.diffusion.prediction_type,
            noise_schedule=cfg.diffusion.noise_schedule,
        )
    elif cfg.diffusion.timestep_type == "continuous" and not hasattr(cfg, 'condition_model'):
        ddpm = ContinuousTimeGaussianDiffusion(
            model=model,
            condition_model=condition_model,
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,
            noise_schedule=cfg.diffusion.noise_schedule,
        )
    elif cfg.diffusion.timestep_type == "continuous" and hasattr(cfg, 'condition_model'):
        ddpm = CondContinuousTimeGaussianDiffusion(
            model=model,
            condition_model=condition_model,
            loss_type=cfg.diffusion.loss_type,
            prediction_type=cfg.diffusion.prediction_type,
            noise_schedule=cfg.diffusion.noise_schedule,
            cond_mode=cfg.diffusion.cond_mode
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion.timestep_type}")

    state_dict = ckpt["ema_weights"] if ema else ckpt["weights"]
    ddpm.load_state_dict(state_dict)
    ddpm.eval()
    ddpm.to(device)

    if compile:
        ddpm.model = torch.compile(ddpm.model)

    lidar_utils = LiDARUtility(
        resolution=cfg.data.resolution,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=ddpm.model.coords,
    )
    lidar_utils.eval()
    lidar_utils.to(device)

    if show_info:
        print(
            *[
                f"resolution: {model.resolution}",
                f"model: {model.__class__.__name__}",
                f"ddpm: {ddpm.__class__.__name__}",
                f'#steps:  {ckpt["global_step"]:,}',
                f"#params: {count_parameters(ddpm):,}",
            ],
            sep="\n",
        )

    cfg.data.split = split
    dataset = all_datasets[cfg.data.dataset](cfg.data)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size_train,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=dataset.collate_fn if getattr(cfg.data, 'custom_collate_fn', False) else None
    )

    return dataset, ddpm, lidar_utils, cfg

def load_model_duffusion_training(cfg: object):

    in_channels = [0, 0]
    if cfg.data.train_depth:
        in_channels[0] = 1
    if cfg.data.train_reflectance:
        in_channels[1] = 1
    in_channels = sum(in_channels)
    if hasattr(cfg, 'condition_model') and getattr(cfg.diffusion, 'cond_mode', None) == 'concat':
        in_channels += cfg.condition_model.params['out_channels']

    if 'dit' in cfg.model.architecture:
        model = __all_dits__[cfg.model.architecture](in_channels=in_channels,
                                                resolution=cfg.data.resolution,
                                                **cfg.model.params)
    else:
        model = __all_unets__[cfg.model.architecture](in_channels=in_channels,
                                                resolution=cfg.data.resolution,
                                                **cfg.model.params)
    
    if "spherical" in cfg.data.projection:
        model.coords = get_linear_ray_angles(H=cfg.data.resolution[0], W=cfg.data.resolution[1], fov_up=cfg.data.fov_up, fov_down=cfg.data.fov_down)
    elif "unfolding" in cfg.data.projection:
        model.coords = F.interpolate(
            torch.load(f"data/{cfg.data.dataset}/unfolding_angles.pth"),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
    else:
        raise ValueError(f"Unknown: {cfg.data.projection}")

    if hasattr(cfg, 'condition_model'):
        condition_model = __all_unets__[cfg.condition_model.architecture](**cfg.condition_model.params)
    else:
        condition_model = nn.Identity()

    if cfg.diffusion.timestep_type == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            model=model,
            loss_type=cfg.diffusion.loss_type,
            num_training_steps=cfg.diffusion.num_training_steps,
            prediction_type=cfg.diffusion.prediction_type,
            noise_schedule=cfg.diffusion.noise_schedule,
        )
    elif cfg.diffusion.timestep_type == "continuous" and not hasattr(cfg, 'condition_model'):
        ddpm = ContinuousTimeGaussianDiffusion(
            model=model,
            condition_model=condition_model,
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,
            noise_schedule=cfg.diffusion.noise_schedule,
        )
    elif cfg.diffusion.timestep_type == "continuous" and hasattr(cfg, 'condition_model'):
        ddpm = CondContinuousTimeGaussianDiffusion(
            model=model,
            condition_model=condition_model,
            loss_type=cfg.diffusion.loss_type,
            prediction_type=cfg.diffusion.prediction_type,
            noise_schedule=cfg.diffusion.noise_schedule,
            cond_mode=getattr(cfg.diffusion, 'cond_mode', None),
            w_loss_weight=getattr(cfg.diffusion, 'w_loss_weight', False)
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion.timestep_type}")

    lidar_utils = LiDARUtility(
        resolution=cfg.data.resolution,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=ddpm.model.coords,
    )
    lidar_utils.eval()
    
    ckpt_path = cfg.resume
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["ema_weights"]
        ddpm.load_state_dict(state_dict)
        ddpm.eval()

        return ddpm, model, lidar_utils, ckpt["global_step"], ckpt['optimizer'], ckpt['lr_scheduler']
    else:
        return ddpm, model, lidar_utils

def load_model_layout_duffusion_training(cfg: object):

    model = __all_unets__[cfg.model.architecture](**cfg.model.params)
    condition_model = __all_unets__[cfg.condition_model.architecture](**cfg.condition_model.params)
    ddpm = CondContinuousLayoutGaussianDiffusion(
        model=model,
        condition_model=condition_model,
        loss_type=cfg.diffusion.loss_type,
        prediction_type=cfg.diffusion.prediction_type,
        noise_schedule=cfg.diffusion.noise_schedule,
        clip_sample=cfg.diffusion.clip_sample
    )

    ckpt_path = cfg.resume
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["ema_weights"]
        ddpm.load_state_dict(state_dict)
        ddpm.eval()

        return ddpm, model, ckpt["global_step"], ckpt['optimizer'], ckpt['lr_scheduler']
    else:
        return ddpm, model

def load_model_object_duffusion_training(cfg: object):

    model = __all_unets__[cfg.model.architecture](**cfg.model.params)
    condition_model = __all_unets__[cfg.condition_model.architecture](**cfg.condition_model.params)
    ddpm = CondContinuousLayoutGaussianDiffusion1D(
        model=model,
        condition_model=condition_model,
        loss_type=cfg.diffusion.loss_type,
        prediction_type=cfg.diffusion.prediction_type,
        noise_schedule=cfg.diffusion.noise_schedule,
        clip_sample=cfg.diffusion.clip_sample
    )

    ckpt_path = cfg.resume
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["ema_weights"]
        ddpm.load_state_dict(state_dict)
        ddpm.eval()

        return ddpm, model, ckpt["global_step"], ckpt['optimizer'], ckpt['lr_scheduler']
    else:
        return ddpm, model

def load_model_flow_training(cfg: object):

    in_channels = [0, 0]
    if cfg.data.train_depth:
        in_channels[0] = 1
    if cfg.data.train_reflectance:
        in_channels[1] = 1
    in_channels = sum(in_channels)

    model = __all_unets__[cfg.model.architecture](in_channels=in_channels,
                                              resolution=cfg.data.resolution,
                                              **cfg.model.params)

    if "spherical" in cfg.data.projection:
        model.coords = get_linear_ray_angles(H=cfg.data.resolution[0], W=cfg.data.resolution[1], fov_up=cfg.data.fov_up, fov_down=cfg.data.fov_down)
    elif "unfolding" in cfg.data.projection:
        model.coords = F.interpolate(
            torch.load(f"data/{cfg.data.dataset}/unfolding_angles.pth"),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
    else:
        raise ValueError(f"Unknown: {cfg.data.projection}")

    # if hasattr(cfg, 'condition_model'):
    #     condition_model = __all_unets__[cfg.condition_model.architecture](**cfg.condition_model.params)
    # else:
    #     condition_model = nn.Identity()

    # if cfg.flow.flow_type == "mean":
    flow = __all_flows__[cfg.flow.flow_type](
        model=model,
        channels = cfg.flow.channels,
        image_size=cfg.data.resolution,
        normalizer=cfg.flow.normalizer,
        time_dist=cfg.flow.time_dist,
        flow_ratio=cfg.flow.flow_ratio,
        cfg_ratio=cfg.flow.cfg_ratio,
        cfg_scale=cfg.flow.cfg_scale,
        jvp_api=cfg.flow.jvp_api
    )

    # else:
    #     raise ValueError(f"Unknown: {cfg.diffusion.timestep_type}")

    lidar_utils = LiDARUtility(
        resolution=cfg.data.resolution,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=flow.model.coords,
    )
    lidar_utils.eval()
    
    ckpt_path = getattr(cfg, 'resume', None)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["ema_weights"]
        flow.load_state_dict(state_dict)
        flow.eval()

        return flow, model, lidar_utils, ckpt["global_step"], ckpt['optimizer'], ckpt['lr_scheduler']
    else:
        return flow, model, lidar_utils

def setup_rng(seeds: list[int], device: torch.device | str):
    return [torch.Generator(device=device).manual_seed(i) for i in seeds]
