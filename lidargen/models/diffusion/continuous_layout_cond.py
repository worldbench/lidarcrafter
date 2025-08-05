import einops
import torch
from torch.cuda.amp import autocast
from torch.special import expm1
from tqdm.auto import tqdm
from typing import Literal, List
from .continuous_time_cond import CondContinuousTimeGaussianDiffusion, _log_snr_to_alpha_sigma

class CondContinuousLayoutGaussianDiffusion(CondContinuousTimeGaussianDiffusion):

    @torch.no_grad()
    def get_scenegraph_input(self, batch):
        enc_objs = batch['encoder']['objs']
        enc_triples = batch['encoder']['tripltes']
        encoded_enc_text_feat = None
        encoded_enc_rel_feat = None
        encoded_dec_text_feat = None
        encoded_dec_rel_feat = None
        encoded_enc_text_feat = batch['encoder']['text_feats'].to(self.device)
        encoded_enc_rel_feat = batch['encoder']['rel_feats'].to(self.device)
        encoded_dec_text_feat = batch['decoder']['text_feats'].to(self.device)
        encoded_dec_rel_feat = batch['decoder']['rel_feats'].to(self.device)

        dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = batch['decoder']['objs'], \
                                                                                        batch['decoder']['tripltes'], \
                                                                                        batch['decoder']['boxes'], \
                                                                                        batch['decoder']['obj_to_scene'], \
                                                                                        batch['decoder']['triple_to_scene']

        if 'feats' in batch['decoder']:
            encoded_dec_f = batch['decoder']['feats']
            encoded_dec_f = encoded_dec_f.to(self.device)

        # changed nodes
        missing_nodes = batch['missing_nodes']
        manipulated_nodes = batch['manipulated_subs'] + batch['manipulated_objs']

        enc_objs, enc_triples = enc_objs.to(self.device), enc_triples.to(self.device)
        dec_objs, dec_triples, dec_tight_boxes = dec_objs.to(self.device), dec_triples.to(self.device), dec_tight_boxes.to(self.device)
        return enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, dec_objs, dec_triples, dec_tight_boxes,\
                encoded_dec_text_feat, encoded_dec_rel_feat, dec_objs_to_scene, dec_triples_to_scene, missing_nodes, manipulated_nodes

    @autocast(enabled=False)
    def q_step_from_x_0(self, x_0, step_t, rng=None):
        # forward diffusion process q(zt|x0) where 0<t<1
        noise = self.randn_like(x_0, rng=rng)
        log_snr = self.log_snr(step_t)
        alpha, sigma = _log_snr_to_alpha_sigma(log_snr)
        alpha.squeeze_(-1).squeeze_(-1)
        sigma.squeeze_(-1).squeeze_(-1)
        x_t = x_0 * alpha + noise * sigma
        return x_t, noise

    def sample_timesteps(self, batch_size: int, sample_ids, device: torch.device) -> torch.Tensor:
        # continuous timesteps
        unique_scenes, inv_idx = torch.unique(sample_ids, return_inverse=True)
        t = torch.rand(unique_scenes.shape[0], device=device, dtype=torch.float32)
        t = t[inv_idx]
        return t

    def prepare_df_input(self, triples, obj_embed, relation_cond, scene_ids=None, obj_boxes=None):
        diff_dict = {'preds': triples, 'box': obj_boxes, 'uc_b': obj_embed,
                     'c_b': relation_cond, "obj_id_to_scene": scene_ids}
        return diff_dict

    def get_network_condition(self, steps=None, input_dict=None, only_custom_condition=False):
        if only_custom_condition:
            latent_obj_vecs, obj_embed_ = self.condition_model(*input_dict['scenegraph_input'])
            other_condition = self.prepare_df_input(input_dict['scenegraph_input'][5], obj_embed_, obj_boxes=input_dict['x_0'], relation_cond=latent_obj_vecs, scene_ids=input_dict['scenegraph_input'][9])
            return dict(other_condition=other_condition)
        time_condition = self.log_snr(steps)[:, 0, 0, 0]
        latent_obj_vecs, obj_embed_ = self.condition_model(*input_dict['scenegraph_input'])
        other_condition = self.prepare_df_input(input_dict['scenegraph_input'][5], obj_embed_, obj_boxes=input_dict['x_0'], relation_cond=latent_obj_vecs, scene_ids=input_dict['scenegraph_input'][9])
        return dict(time_condition=time_condition, other_condition=other_condition)
    
    def p_loss(
        self,
        input_dict: dict,
        steps: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # shared in continuous/discrete versions
        x_0 = input_dict['x_0']
        loss_mask = torch.ones_like(x_0) if loss_mask is None else loss_mask
        x_t, noise = self.q_step_from_x_0(x_0, steps)
        condition = self.get_network_condition(steps, input_dict)
        prediction = self.model(x_t, condition)
        target = self.get_target(x_0, steps, noise)
        loss = self.criterion(prediction, target)  # (B,C,H,W)
        loss = einops.reduce(loss * loss_mask, "B ... -> B ()", "sum")
        loss_mask = einops.reduce(loss_mask, "B ... -> B ()", "sum")
        loss = loss / loss_mask.add(1e-8)  # (B,)
        loss = (loss * self.get_loss_weight(steps)).mean()
        return loss
    
    @torch.inference_mode()
    def p_step(
        self,
        x_t: torch.Tensor,
        condition_dict: dict,
        step_t: torch.Tensor,
        step_s: torch.Tensor,
        rng: List[torch.Generator] | torch.Generator | None = None,
        mode: Literal["ddpm", "ddim"] = "ddpm",
        ddim_eta: float = 0.0,
    ) -> torch.Tensor:
        # reverse diffusion process p(zs|zt) where 0<s<t<1
        log_snr_t = self.log_snr(step_t)
        log_snr_s = self.log_snr(step_s)
        alpha_t, sigma_t = _log_snr_to_alpha_sigma(log_snr_t)
        alpha_s, sigma_s = _log_snr_to_alpha_sigma(log_snr_s)
        alpha_t.squeeze_(-1).squeeze_(-1)
        alpha_s.squeeze_(-1).squeeze_(-1)
        sigma_t.squeeze_(-1).squeeze_(-1)
        sigma_s.squeeze_(-1).squeeze_(-1)
        condition_dict.update(dict(time_condition=log_snr_t[:, 0, 0, 0]))
        prediction = self.model(x_t, condition_dict)
        if self.objective == "eps":
            x_0 = (x_t - sigma_t * prediction) / alpha_t
        elif self.objective == "v":
            x_0 = alpha_t * x_t - sigma_t * prediction
        elif self.objective == "x_0":
            x_0 = prediction
        else:
            raise ValueError(f"invalid objective {self.objective}")
        if self.clip_sample:
            x_0.clamp_(-self.clip_sample_range, self.clip_sample_range)
        if mode == "ddpm":
            c = -expm1(log_snr_t - log_snr_s)
            c.squeeze_(-1).squeeze_(-1)
            mean = alpha_s * (x_t * (1 - c) / alpha_t + c * x_0)
            std = sigma_s * c.sqrt()
            noise = self.randn_like(x_t, rng=rng)
            x_s = mean + std * noise
        elif mode == "ddim":
            c_1 = ddim_eta * sigma_s / sigma_t * (1 - alpha_t**2 / alpha_s**2).sqrt()
            c_2 = (1 - alpha_s**2 - c_1**2).sqrt()
            eps = (x_t - alpha_t * x_0) / sigma_t
            noise = self.randn_like(x_t, rng=rng)
            x_s = alpha_s * x_0 + c_1 * noise + c_2 * eps
        else:
            raise ValueError(f"invalid mode {mode}")
        return x_s

    def forward(
        self,
        input_dict: dict,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # shared in continuous/discrete versions
        scegraph_input_dict = self.get_scenegraph_input(input_dict['scenegraph_input'])
        x_0 = scegraph_input_dict[6][:,:20]
        loss_mask = scegraph_input_dict[6][:,20:]
        input_dict['x_0'] = x_0
        input_dict['scenegraph_input'] = scegraph_input_dict
        scene_ids = scegraph_input_dict[9]
        steps = self.sample_timesteps(x_0.shape[0], scene_ids, x_0.device)
        loss = self.p_loss(input_dict, steps, loss_mask) # TODO: add moiu loss
        return loss
    
    @torch.inference_mode()
    def sample(
        self,
        batch_dict: dict,
        num_steps: int,
        progress: bool = True,
        rng: list[torch.Generator] | torch.Generator | None = None,
        return_all: bool = False,
        mode: Literal["ddpm", "ddim"] = "ddpm",
        ddim_eta: float = 0.0,
    ):
        scegraph_input_dict = self.get_scenegraph_input(batch_dict['scenegraph_input'])
        x_0 = scegraph_input_dict[6][:,:20]
        batch_dict['x_0'] = x_0
        batch_dict['scenegraph_input'] = scegraph_input_dict
        batch_size = x_0.shape[0]
        x = self.randn(batch_size, *self.sampling_shape, rng=rng, device=self.device).squeeze(-1)
        condition_dict = self.get_network_condition(input_dict=batch_dict, only_custom_condition=True)
        if return_all:
            out = [x]
        steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        steps = steps[None].repeat_interleave(batch_size, dim=0)
        p_step_kwargs = dict(rng=rng, mode=mode, ddim_eta=ddim_eta)
        tqdm_kwargs = dict(desc="sampling", leave=False, disable=not progress)
        for i in tqdm(range(num_steps), **tqdm_kwargs):
            step_t = steps[:, i]
            step_s = steps[:, i + 1]
            x = self.p_step(x, condition_dict, step_t, step_s, **p_step_kwargs)
            if return_all:
                out.append(x)
        return torch.stack(out) if return_all else x