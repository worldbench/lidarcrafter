import torch
from lidargen.utils import inference
from lidargen.utils import common
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets

class Layout_Sampler:
    def __init__(self, cfg, ckpt_path):
        self.cfg = __all__[cfg]()
        self.cfg.resume = ckpt_path
        self.init_sampler()

    def init_sampler(self):
        self.cfg.data.split = 'val'
        dataset = all_datasets[self.cfg.data.dataset](self.cfg.data)
        self.cfg.condition_model.params['vocab'] = dataset.scene_graph_assigner.vocab
        self.ddpm, _, _, _, _ = inference.load_model_layout_duffusion_training(self.cfg)
        self.ddpm.eval()
        self.ddpm.to('cuda')
        self.dataset = dataset

    def sample(self, num_steps=1024, sample_index=None):
        batch_dict = self.dataset.__getitem__(sample_index)
        batch_dict.pop('gt_fut_trajs')
        batch_dict.pop('gt_fut_masks')
        batch_dict.pop('gt_fut_states')
        collate_fn = self.dataset.collate_fn if getattr(self.cfg.data, 'custom_collate_fn', False) else None
        if collate_fn is not None:
            batch_dict = collate_fn([batch_dict])
        batch_dict = common.to_device(batch_dict, 'cuda')

        boxes_3d_traj = self.ddpm.sample(
            batch_dict = batch_dict,
            num_steps=num_steps,
            mode='ddpm',
            return_all=False,
        )
        unscaled_boxes, boxes_trajs = self.dataset.unscale_boxes_3d(boxes_3d_traj)
        return batch_dict['gt_names'][0], unscaled_boxes, boxes_trajs