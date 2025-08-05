from typing import Literal, Tuple
from pydantic.dataclasses import dataclass
from dataclasses import field

@dataclass
class ModelConfig:
    architecture: str = "point_unet"
    params: dict = field(
        default_factory=lambda: {
            'point_dim': 4,
            'cond_dims': 768,
        }
    )

@dataclass
class ConditionModelConfig:
    architecture: str = "object_gen_encoder"
    params: dict = field(
        default_factory=lambda: {
            'num_class': 8,
        }
    )

@dataclass
class DiffusionConfig:
    num_training_steps: int | None = None
    num_sampling_steps: int = 1024
    prediction_type: Literal["eps", "v", "x_0"] = "eps"
    loss_type: str = "l2"
    noise_schedule: str = "cosine"
    timestep_type: Literal["continuous", "discrete"] = "continuous"
    clip_sample: bool = False

@dataclass
class TrainingConfig:
    batch_size_train: int = 2
    batch_size_eval: int = 8
    num_workers: int = 4
    num_steps: int = 1_000_000
    steps_save_image: int = 5_000
    steps_save_model: int = 100_000
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
    task: str = "object_generation"
    dataset: Literal["kitti_raw", "kitti_360", "nuscenes", "nuscenes-object"] = "nuscenes-object"
    class_names: Tuple[str, str, str, str, str, str, str, str] = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian')
    custom_collate_fn: bool = True
    data_root = "../data/nuscenes"
    pkl_path = "../data/infos/nuscenes_dbinfos_10sweeps_withvelo.pkl"
    num_samples: int = 1024

@dataclass
class NUSC_Object_Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    condition_model: ConditionModelConfig = ConditionModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
