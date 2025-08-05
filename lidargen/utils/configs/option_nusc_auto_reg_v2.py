'''
1. Do not use the intensity channel in the layout diffusion
2. Multi-history condition
3. Add noise while training
'''

from typing import Literal, Tuple
from pydantic.dataclasses import dataclass
from dataclasses import field

@dataclass
class ModelConfig:
    architecture: str = "layout_unet_v1"
    params: dict = field(
        default_factory=lambda: {
            'image_size': 32,
            'use_fp16': False,
            'use_scale_shift_norm': True,
            'out_channels': 2,
            'model_channels': 64,
            'encoder_channels': 64,
            'num_head_channels': 32,
            'num_heads': -1,
            'num_heads_upsample': -1,
            'num_res_blocks': 2,
            'num_attention_blocks': 1,
            'resblock_updown': True,
            'attention_ds': [4, 8],
            'channel_mult': [1, 2, 4, 8],
            'dropout': 0.1,
            'use_checkpoint': False,
            'use_positional_embedding_for_attention': True,
            'attention_block_type': 'ObjectAwareCrossAttention',
        }
    )

@dataclass
class ConditionModelConfig:
    architecture: str = "layout_encoder"
    params: dict = field(
        default_factory=lambda: {
            'feature_map_size': [32, 1024],
            'used_condition_types': ['obj_class', 'obj_bbox', 'is_valid_obj'],
            'layout_length': 13,
            'num_classes_for_layout_object': 9,  # 8 + 1
            'mask_size_for_layout_object': 32,
            'hidden_dim': 64,
            'output_dim': 256,  # model_channels x 4
            'num_layers': 6,
            'num_heads': 4,
            'use_final_ln': True,
            'use_positional_embedding': False,
            'not_use_layout_fusion_module': False,
            'resolution_to_attention': [4, 8],
            'use_key_padding_mask': False,
            'out_channels': 11
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
    cond_mode: str = "concat"
    # w_loss_weight: bool = True

@dataclass
class TrainingConfig:
    batch_size_train: int = 2
    batch_size_eval: int = 8
    num_workers: int = 4
    num_steps: int = 500_000
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
    task: str = "autoregressive_generation"
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
class NUSC_Auto_Reg_V2_Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    condition_model: ConditionModelConfig = ConditionModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
