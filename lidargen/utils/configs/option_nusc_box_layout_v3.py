'''
1. With data arguments
'''

from typing import Literal, Tuple, List, Union
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
            'out_channels': 10
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

# —— 1. gt_sampling  —— #
@dataclass
class GtSamplingPrepare:
    filter_by_min_points: List[str] = field(default_factory=lambda: [
        'car:5','truck:5','construction_vehicle:5','bus:5','trailer:5',
        'barrier:5','motorcycle:5','bicycle:5','pedestrian:5'
    ])

@dataclass
class GtSamplingConfig:
    name: Literal["gt_sampling"] = "gt_sampling"
    db_info_path: List[str] = field(default_factory=lambda: [
        "nuscenes_dbinfos_10sweeps_withvelo.pkl"
    ])
    prepare: GtSamplingPrepare = field(default_factory=GtSamplingPrepare)
    sample_groups: List[str] = field(default_factory=lambda: [
        # 'car:2','truck:3','construction_vehicle:7','bus:4','trailer:6',
        # 'barrier:2','motorcycle:6','bicycle:6','pedestrian:2'
        'car:4'
    ])
    num_point_features: int = 5
    database_with_fakelidar: bool = False
    remove_extra_width: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    limit_whole_scene: bool = True

# —— 2. augmenters —— #
@dataclass
class RandomWorldFlipConfig:
    name: Literal["random_world_flip"] = "random_world_flip"
    along_axis_list: List[str] = field(default_factory=lambda: ['x', 'y'])

@dataclass
class RandomWorldRotationConfig:
    name: Literal["random_world_rotation"] = "random_world_rotation"
    world_rot_angle: Tuple[float, float] = (-0.3925, 0.3925)

@dataclass
class RandomWorldScalingConfig:
    name: Literal["random_world_scaling"] = "random_world_scaling"
    world_scale_range: Tuple[float, float] = (0.95, 1.05)

# —— 3. merge —— #
AugConfig = Union[
    GtSamplingConfig,
    RandomWorldFlipConfig,
    RandomWorldRotationConfig,
    RandomWorldScalingConfig,
]

# —— 4. —— #
@dataclass
class DataAugmentorConfig:
    disable_aug_list: List[str] = field(default_factory=lambda: ['placeholder', 'random_world_rotation', 'random_world_scaling', 'random_world_flip'])
    aug_config_list: List[AugConfig] = field(default_factory=lambda: [
        GtSamplingConfig(),
        RandomWorldFlipConfig(),
        RandomWorldRotationConfig(),
        RandomWorldScalingConfig(),
    ])

@dataclass
class DataConfig:
    dataset: Literal["kitti_raw", "kitti_360", "nuscenes"] = "nuscenes"
    task: str = "layout_cond"
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
    data_augmentor: DataAugmentorConfig = field(default_factory=DataAugmentorConfig)

@dataclass
class NUSC_Box_Layout_V3_Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    condition_model: ConditionModelConfig = ConditionModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
