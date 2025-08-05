import torch
import torch.nn.functional as F
from loguru import logger
from lidargen.utils import inference
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets

class Lidargen_Sampler:
    def __init__(self, cfg, ckpt_path, conditioned_sample=False):
        self.cfg_name = cfg
        self.cfg = __all__[cfg]()
        self.cfg.resume = ckpt_path
        self.conditioned_sample = conditioned_sample
        self.init_sampler()

    def init_sampler(self):
        if self.conditioned_sample:
            self.cfg.data.split = 'val'
            self.cfg.data.delete_ground = True # TODO: Just for temporary use
            self.dataset = all_datasets[self.cfg.data.dataset](self.cfg.data)
            self.ddpm, _, self.lidar_utils, _, _, _ = inference.load_model_duffusion_training(self.cfg)
            self.ddpm.eval()
            self.lidar_utils.eval()
            self.ddpm.to('cuda')
            self.lidar_utils.to('cuda')
        else:
            self.ddpm, _, self.lidar_utils, _, _, _ = inference.load_model_duffusion_training(self.cfg)
            self.dataset = None
            self.ddpm.eval()
            self.lidar_utils.eval()
            self.ddpm.to('cuda')
            self.lidar_utils.to('cuda')
        logger.info(f"Loaded diffusion model {self.cfg_name}")

    def preprocess(self, batch):
        x = []
        if self.cfg.data.train_depth:
            x += [self.lidar_utils.convert_depth(batch["depth"])]
        if self.cfg.data.train_reflectance:
            x += [batch["reflectance"]]
        x = torch.cat(x, dim=1)
        x = self.lidar_utils.normalize(x)
        x = F.interpolate(
            x.to('cuda'),
            size=self.cfg.data.resolution,
            mode="nearest-exact",
        )
        return x
    
    def preprocess_prev_cond(self, prev_cond):
        x = []
        reflectance = prev_cond[:,3,...]/255
        depth = prev_cond[:,-2,...]

        if self.cfg.data.train_depth:
            x += [self.lidar_utils.convert_depth(depth.unsqueeze(1))]
        if self.cfg.data.train_reflectance:
            x += [reflectance.unsqueeze(1)]
        x = torch.cat(x, dim=1)
        x = self.lidar_utils.normalize(x)
        x = F.interpolate(
            x.to('cuda'),
            size=self.cfg.data.resolution,
            mode="nearest-exact",
        )
        prev_labels = prev_cond[:,4,...].long()
        one_hot = F.one_hot(prev_labels, num_classes=len(list(self.cfg.data.class_names))+1).permute(0, 3, 1, 2)
        x = torch.cat((x, one_hot.float()), dim=1)
        return x

    def preprocess_condition_mask(self, batch):
        x = []
        condition_mask = batch['condition_mask'] # [B, 2, H, W]: semantic and depth
        # semantic
        curr_labels = condition_mask[:,0,...].long()
        one_hot = F.one_hot(curr_labels, num_classes=len(list(self.cfg.data.class_names))+1).permute(0, 3, 1, 2)
        x+= [one_hot.float()]
        # depth
        depth = self.lidar_utils.convert_depth(condition_mask[:,1,...].unsqueeze(1))
        x+= [depth]
        x = torch.cat(x, dim=1)
        return x

    def preprocess_autoregressive_cond(self, autoregressive_cond):
        x = []
        depth = autoregressive_cond[:, 0]
        reflectance = autoregressive_cond[:, 1]

        if self.cfg.data.train_depth:
            x += [self.lidar_utils.convert_depth(depth.unsqueeze(1))]
        if self.cfg.data.train_reflectance and self.cfg_name != 'nuscenes-auto-reg-v2':
            x += [reflectance.unsqueeze(1)]
        x = torch.cat(x, dim=1)
        x = self.lidar_utils.normalize(x)
        x = F.interpolate(
            x.to('cuda'),
            size=self.cfg.data.resolution,
            mode="nearest-exact",
        )
        return x

    def prepare_batch(self, batch_dict):
        collate_fn = self.dataset.collate_fn if getattr(self.cfg.data, 'custom_collate_fn', False) else None
        if collate_fn is not None:
            batch_dict = collate_fn([batch_dict])
        for key in batch_dict:
            if isinstance(batch_dict[key], torch.Tensor):
                batch_dict[key] = batch_dict[key].to('cuda')

        # if self.cfg. TODO: weather to use the collate_fn
        # x_0 = self.preprocess(batch_dict)
        # batch_dict['x_0'] = x_0
        if 'prev_cond' in batch_dict:
            prev_cond = self.preprocess_prev_cond(batch_dict['prev_cond'])
            batch_dict['cond'] = prev_cond

        if 'condition_mask' in batch_dict:
            condition_mask = self.preprocess_condition_mask(batch_dict)
            batch_dict['concat_cond'] = condition_mask

        if 'autoregressive_cond' in batch_dict:
            autoregressive_cond = self.preprocess_autoregressive_cond(batch_dict['autoregressive_cond'])
            batch_dict['autoregressive_cond'] = autoregressive_cond

        return batch_dict

    def prepare_inpaint(self, batch_dict, inpaint_cond_dict):
        if "inpaint_cond" in inpaint_cond_dict:
            if 'condition_mask' in inpaint_cond_dict:
                new_condition_mask = batch_dict['condition_mask'][:,0,...]>0
                last_condition_mask = inpaint_cond_dict['condition_mask']
                # keep the same condition mask
                mask = (new_condition_mask == last_condition_mask).float()
                x_in = inpaint_cond_dict['inpaint_cond']

            elif 'mask' in inpaint_cond_dict:
                x_in = inpaint_cond_dict['inpaint_cond']
                mask = inpaint_cond_dict['mask'].float()

        else:
            collate_fn = self.dataset.collate_fn if getattr(self.cfg.data, 'custom_collate_fn', False) else None
            if collate_fn is not None:
                inpaint_cond_dict = collate_fn([inpaint_cond_dict])
            for key in inpaint_cond_dict:
                if isinstance(inpaint_cond_dict[key], torch.Tensor):
                    inpaint_cond_dict[key] = inpaint_cond_dict[key].to('cuda')

            x_in = self.preprocess(inpaint_cond_dict)
            mask = inpaint_cond_dict['mask']
            # mask[0,0,16:]=0
            # save condition
            # import numpy as np
            # x_in = self.lidar_utils.denormalize(x_in)
            # x_in[:, [0]] = self.lidar_utils.revert_depth(x_in[:, [0]])
            # x_in[:, [0]] = x_in[:, [0]]* mask[:, [0]]  # apply mask to depth
            # points = self.lidar_utils.to_xyz(x_in[:, [0]])
            # points = points[0].reshape(3,-1).permute(1,0).detach().cpu().numpy()
            # np.savetxt('../tools/ALTest/temp/inpaint_cond.txt', points, fmt='%.6f')
            

        return x_in.to('cuda'), mask.to('cuda')

    def sample(self, num_steps=1024, sample_index=None, batch_dict=None):
        if self.conditioned_sample:
            # confirm at least one of sample_index or batch_dict is provided
            assert sample_index is not None or batch_dict is not None, \
                "Either sample_index or batch_dict must be provided for conditioned sampling."
            if batch_dict is None:
                batch_dict = self.dataset.__getitem__(sample_index)
            batch_dict = self.prepare_batch(batch_dict)

            xs = self.ddpm.sample(
                batch_dict=batch_dict,
                batch_size=1,
                num_steps=num_steps,
                mode="ddpm",
                return_all=False,
            ).clamp(-1, 1)

            # inpaint cond
            inpaint_cond_dict = dict(
                inpaint_cond = xs,
                condition_mask = batch_dict['condition_mask'][:,0,...]>0
            )

        else:
            xs = self.ddpm.sample(
                batch_size=1,
                num_steps=num_steps,
                mode="ddpm",
                return_all=False,
            ).clamp(-1, 1)
            inpaint_cond_dict = None

        xs = self.lidar_utils.denormalize(xs)
        xs[:, [0]] = self.lidar_utils.revert_depth(xs[:, [0]])
        points = self.lidar_utils.to_xyz(xs[:, [0]]) # [1, 3, 32, 1024]
        intensity = xs[:, [1]] # [1, 1, 32, 1024]
        points = torch.cat((points, intensity), dim=1)
        points = points[0].reshape(4,-1).permute(1,0).detach().cpu().numpy()
        return points, xs[:, [0]], inpaint_cond_dict
    
    def inpaint(self, num_steps=1024, batch_dict=None, inpaint_cond_dict=None):
        assert self.conditioned_sample, \
            "Inpainting is only supported for conditioned sampling."
        
        batch_dict = self.prepare_batch(batch_dict)
        x_in, mask = self.prepare_inpaint(batch_dict, inpaint_cond_dict)

        xs = self.ddpm.inpaint(
            known=x_in,
            mask=mask,
            batch_dict=batch_dict,
            num_steps=num_steps,
            return_all=False,
        ).clamp(-1, 1)

        # inpaint cond
        inpaint_cond_dict = dict(
            inpaint_cond = xs,
            condition_mask = batch_dict['condition_mask'][:,0,...]>0
        )

        xs = self.lidar_utils.denormalize(xs)
        xs[:, [0]] = self.lidar_utils.revert_depth(xs[:, [0]])
        points = self.lidar_utils.to_xyz(xs[:, [0]]) # [1, 3, 32, 1024]
        intensity = xs[:, [1]] # [1, 1, 32, 1024]
        points = torch.cat((points, intensity), dim=1)
        points = points[0].reshape(4,-1).permute(1,0).detach().cpu().numpy()
        return points, xs[:, [0]], inpaint_cond_dict