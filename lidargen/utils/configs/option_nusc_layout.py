from typing import Literal, Tuple
from pydantic.dataclasses import dataclass
from dataclasses import field

@dataclass
class ModelConfig:
    architecture: str = "unet_1d"
    params: dict = field(
        default_factory=lambda: {
            'dims': 1,
            'in_channels': 20, # 8 + 12
            'out_channels': 20,
            'model_channels': 512,
            'channel_mult': [1,1,1,1],
            'num_res_blocks': 2,
            'attention_resolutions': [4,2],
            'num_heads': 8,
            'use_spatial_transformer': True,
            'transformer_depth': 1,
            'conditioning_key': 'crossattn',
            'concat_dim': 1280,
            'crossattn_dim': 1280,
            'use_checkpoint': True,
            'enable_t_emb': True,
        }
    )

@dataclass
class ConditionModelConfig:
    architecture: str = "scene_graph"
    params: dict = field(
        default_factory=lambda: {
            'embedding_dim': 64,
            'gconv_pooling': 'avg',
            'gconv_num_layers': 5,
            'mlp_normalization': 'batch',
            'separated': True,
            'replace_latent': True,
            'residual': True,
            'use_angles': True,
            'use_clip': True,
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
    num_steps: int = 300_000
    steps_save_image: int = 5_000
    steps_save_model: int = 50_000
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
    dataset: Literal["kitti_raw", "kitti_360", "nuscenes"] = "nuscenes"
    task: str = "layout_generation"
    with_object: bool = False
    class_names: Tuple[str, str, str, str, str, str, str, str] = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian')
    custom_collate_fn: bool = True
    data_root = "../data/nuscenes"
    pkl_path = "../data/infos/nuscenes_infos_lidargen_train.pkl"

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
class NUSC_Layout_Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    condition_model: ConditionModelConfig = ConditionModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
