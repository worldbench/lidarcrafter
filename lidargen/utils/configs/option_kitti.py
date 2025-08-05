from typing import Literal, Tuple

from pydantic.dataclasses import dataclass


@dataclass
class ModelConfig:
    architecture: str = "efficient_unet"
    base_channels: int = 64
    temb_channels: int | None = None
    channel_multiplier: Tuple[int, int, int, int] = (1, 2, 4, 8)
    num_residual_blocks: Tuple[int, int, int, int] = (3, 3, 3, 3)
    gn_num_groups: int = 32 // 4
    gn_eps: float = 1e-6
    attn_num_heads: int = 8
    coords_encoding: Literal[
        "spherical_harmonics", "polar_coordinates", "fourier_features", None
    ] = "fourier_features"
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    num_training_steps: int | None = None
    num_sampling_steps: int = 1024
    prediction_type: Literal["eps", "v", "x_0"] = "eps"
    loss_type: str = "l2"
    noise_schedule: str = "cosine"
    timestep_type: Literal["continuous", "discrete"] = "continuous"


@dataclass
class TrainingConfig:
    batch_size_train: int = 2
    batch_size_eval: int = 8
    num_workers: int = 4
    num_steps: int = 300_000
    steps_save_image: int = 5_000
    steps_save_model: int = 10_000
    gradient_accumulation_steps: int = 1
    lr: float = 1e-4
    lr_warmup_steps: int = 10_000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    ema_decay: float = 0.995
    ema_update_every: int = 10
    mixed_precision: str = "fp16"
    dynamo_backend: str = "inductor"
    output_dir: str = "logs/diffusion"
    seed: int = 0


@dataclass
class DataConfig:
    dataset: Literal["kitti_raw", "kitti_360"] = "kitti_360"
    depth_format: Literal["log_depth", "inverse_depth", "depth"] = "log_depth"
    projection: Literal[
        "unfolding-2048",
        "spherical-2048",
        "unfolding-1024",
        "spherical-1024",
    ] = "spherical-1024"
    train_depth: bool = True
    train_reflectance: bool = True
    resolution: Tuple[int, int] = (64, 1024)
    min_depth = 1.45
    max_depth = 80.0


@dataclass
class KITTI_Config_:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
