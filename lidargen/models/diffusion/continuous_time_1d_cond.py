import einops
import torch
from torch.cuda.amp import autocast
from torch.special import expm1
from tqdm.auto import tqdm
from typing import Literal, List
from .continuous_time_cond import CondContinuousTimeGaussianDiffusion, _log_snr_to_alpha_sigma

class CondContinuousLayoutGaussianDiffusion1D(CondContinuousTimeGaussianDiffusion):
    def __init__(self, model, condition_model = None, prediction_type = "eps", loss_type = "l2", noise_schedule = "cosine", min_snr_loss_weight = True, min_snr_gamma = 5, sampling_resolution = None, clip_sample = True, clip_sample_range = 1, image_d = None, noise_d_low = None, noise_d_high = None, cond_mode = None):
        super().__init__(model, condition_model, prediction_type, loss_type, noise_schedule, min_snr_loss_weight, min_snr_gamma, sampling_resolution, clip_sample, clip_sample_range, image_d, noise_d_low, noise_d_high, cond_mode)
        self.sampling_shape = (self.sampling_shape[1], self.sampling_shape[0])

    @autocast(enabled=False)
    def q_step_from_x_0(self, x_0, step_t, rng=None):
        # forward diffusion process q(zt|x0) where 0<t<1
        noise = self.randn_like(x_0, rng=rng)
        log_snr = self.log_snr(step_t)
        alpha, sigma = _log_snr_to_alpha_sigma(log_snr)
        alpha.squeeze_(-1)
        sigma.squeeze_(-1)
        x_t = x_0 * alpha + noise * sigma
        return x_t, noise

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
        alpha_t.squeeze_(-1)
        alpha_s.squeeze_(-1)
        sigma_t.squeeze_(-1)
        sigma_s.squeeze_(-1)
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
            c.squeeze_(-1)
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