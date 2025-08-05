import numpy as np
from omegaconf import OmegaConf
from scripts.sample_cond import load_model, custom_to_pcd
from scripts.eval_ae import load_model as load_ae_model
class LiDM_Sampler(object):
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path

    def build_model(self):
        logdir = '/'.join(self.ckpt_path.split('/')[:-1])
        base_configs = f'{logdir}/config.yaml'
        configs = OmegaConf.load(base_configs)
        self.configs = configs
        self.model, _ = load_model(configs, self.ckpt_path)
        self.data_config = {'dataset_config': configs.data.params.dataset, 
                            'aug_config': configs.data.params.aug}
        
    def sample_from_cond(self, batch):
        log_config = {'sample': True, 'ddim_steps': 50,
                      'quantize_denoised': False, 'inpaint': False, 'plot_progressive_rows': False,
                      'plot_diffusion_rows': False, 'dset': 'nuscenes'}
        
        logs = self.model.log_images(batch, N=1, split='val', **log_config)
        recon_points = custom_to_pcd(logs["reconstruction"][0], self.configs)[0].astype(np.float32)
        sample_points = custom_to_pcd(logs["samples"][0], self.configs)[0].astype(np.float32)
        return recon_points, sample_points
    
class AE_Sampler(object):
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path

    def build_model(self):
        logdir = '/'.join(self.ckpt_path.split('/')[:-1])
        base_configs = f'{logdir}/config.yaml'
        configs = OmegaConf.load(base_configs)
        self.configs = configs
        self.model, _ = load_ae_model(configs, self.ckpt_path)
        self.data_config = {'dataset_config': configs.data.params.dataset, 
                            'aug_config': configs.data.params.aug}
        
    def sample_points(self, batch):
        
        logs = self.model.log_images(batch)
        recon_points_s1 = custom_to_pcd(logs["reconstruction_s1"][0], self.configs)[0].astype(np.float32)
        recon_points_s2 = custom_to_pcd(logs["reconstruction_s2"][0], self.configs)[0].astype(np.float32)
        return recon_points_s1, recon_points_s2