from typing import Literal, Tuple
from dataclasses import field
from pydantic.dataclasses import dataclass


@dataclass
class ModelConfig:
    architecture: str = "hdit"
    params: dict = field(
        default_factory=lambda: {
            'base_channels': 128,
            'time_embed_channels': 256,
            'depths': (3, 3, 3, 3),
            'dilation': (1, 1, 1, 1),
            'positional_embedding': "learnable_embedding",
            'ring': True
        })

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
    num_steps: int = 2_560_000
    steps_save_image: int = 5_000
    steps_save_model: int = 100_000
    gradient_accumulation_steps: int = 1
    lr: float = 1e-4
    lr_warmup_steps: int = 80_000
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
    dataset: Literal["kitti_raw", "kitti_360", "nuscenes"] = "nuscenes"
    data_root = "../data/nuscenes"
    depth_format: Literal["log_depth", "inverse_depth", "depth"] = "log_depth"
    scan_unfolding: bool = False
    projection: Literal[
        "unfolding-2048",
        "spherical-2048",
        "unfolding-1024",
        "spherical-1024",
    ] = "spherical-1024"
    train_depth: bool = True
    train_reflectance: bool = True
    resolution: Tuple[int, int] = (32, 1024)
    min_depth = 1.45
    max_depth = 80.0
    fov_up = 10.0
    fov_down = -30.0


@dataclass
class NUSC_HDIT_Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
