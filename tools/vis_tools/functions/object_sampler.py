import torch
from lidargen.utils import inference
from lidargen.utils import common
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets

class Object_Sampler:
    def __init__(self, cfg, ckpt_path):
        self.cfg = __all__[cfg]()
        self.data_cfg = __all__['nuscenes-box-layout']().data
        self.data_cfg.task = 'object_generation'
        self.cfg.resume = ckpt_path
        self.init_sampler()

    def init_sampler(self):
        self.data_cfg.split = 'val'
        dataset = all_datasets[self.data_cfg.dataset](self.data_cfg)
        self.ddpm, _, _, _, _ = inference.load_model_object_duffusion_training(self.cfg)
        self.ddpm.eval()
        self.ddpm.to('cuda')
        self.dataset = dataset

    def sample(self, num_steps=1024, custom_data_dict=None, sample_index=None):
        if custom_data_dict is not None:
            batch_dict = custom_data_dict
        else:
            batch_dict = self.dataset.__getitem__(sample_index)
        collate_fn = self.dataset.collate_fn if getattr(self.cfg.data, 'custom_collate_fn', False) else None
        if collate_fn is not None:
            batch_dict = collate_fn([batch_dict])
        batch_dict = common.to_device(batch_dict, 'cuda')
        fg_encoding_box = batch_dict['fg_encoding_box'].squeeze(0)
        B, _ = fg_encoding_box.shape
        batch_dict['fg_encoding_box'] = fg_encoding_box
        batch_dict['fg_class'] = batch_dict['fg_class'].squeeze(0)
        
        generated_object_points = self.ddpm.sample(
            batch_dict = batch_dict,
            batch_size=B,
            num_steps=num_steps,
            mode='ddpm',
            return_all=False,
        )
        obj_points = self.dataset.unscaled_objs_3d(sample_index, custom_data_dict, generated_object_points.detach().cpu().numpy())
        return obj_points