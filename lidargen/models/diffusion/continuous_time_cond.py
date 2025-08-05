import math
from functools import partial
from typing import List, Literal

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.special import expm1
from tqdm.auto import tqdm
import einops

from . import continuous_time

def _log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def _log_snr_schedule_linear(t: torch.Tensor) -> torch.Tensor:
    return -_log(expm1(1e-4 + 10 * (t**2)))[:, None, None, None]


def _log_snr_schedule_cosine(
    t: torch.Tensor,
    logsnr_min: float = -15,
    logsnr_max: float = 15,
) -> torch.Tensor:
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * _log(torch.tan(t_min + t * (t_max - t_min)))[:, None, None, None]


def _log_snr_schedule_cosine_shifted(
    t: torch.Tensor,
    image_d: float,
    noise_d: float,
    logsnr_min: float = -15,
    logsnr_max: float = 15,
) -> torch.Tensor:
    log_snr = _log_snr_schedule_cosine(t, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
    shift = 2 * math.log(noise_d / image_d)
    return log_snr + shift


def _log_snr_schedule_cosine_interpolated(
    t: torch.Tensor,
    image_d: float,
    noise_d_low: float,
    noise_d_high: float,
    logsnr_min: float = -15,
    logsnr_max: float = 15,
) -> torch.Tensor:
    logsnr_low = _log_snr_schedule_cosine_shifted(
        t, image_d, noise_d_low, logsnr_min, logsnr_max
    )
    logsnr_high = _log_snr_schedule_cosine_shifted(
        t, image_d, noise_d_high, logsnr_min, logsnr_max
    )
    return t * logsnr_low + (1 - t) * logsnr_high


def _log_snr_to_alpha_sigma(log_snr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha, sigma = log_snr.sigmoid().sqrt(), (-log_snr).sigmoid().sqrt()
    return alpha, sigma


class CondContinuousTimeGaussianDiffusion(continuous_time.ContinuousTimeGaussianDiffusion):
    """
    Continuous-time Gaussian diffusion
    https://arxiv.org/pdf/2107.00630.pdf
    """

    def __init__(
        self,
        model: nn.Module,
        condition_model: nn.Module=None,
        prediction_type: Literal["eps", "v", "x_0"] = "eps",
        loss_type: Literal["l2", "l1", "huber"] | nn.Module = "l2",
        noise_schedule: Literal[
            "linear", "cosine", "cosine_shifted", "cosine_interpolated"
        ] = "cosine",
        min_snr_loss_weight: bool = True,
        min_snr_gamma: float = 5.0,
        sampling_resolution: tuple[int, int] | None = None,
        clip_sample: bool = True,
        clip_sample_range: float = 1,
        image_d: float = None,
        noise_d_low: float = None,
        noise_d_high: float = None,
        cond_mode:str = None,
        w_loss_weight: bool = False,
    ):
        super().__init__(
            model=model,
            condition_model=condition_model,
            prediction_type=prediction_type,
            loss_type=loss_type,
            noise_schedule=noise_schedule,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma,
            sampling_resolution=sampling_resolution,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            image_d=image_d,
            noise_d_low=noise_d_low,
            noise_d_high=noise_d_high,
        )
        self.cond_mode = cond_mode
        if self.cond_mode == 'concat':
            self.sampling_shape = (
                self.model.in_channels - condition_model.out_channels,
                *self.sampling_shape[1:],
            )
        self.w_loss_weight = w_loss_weight

    def setup_parameters(self) -> None:
        if self.noise_schedule == "linear":
            self.log_snr = _log_snr_schedule_linear
        elif self.noise_schedule == "cosine":
            self.log_snr = _log_snr_schedule_cosine
        elif self.noise_schedule == "cosine_shifted":
            assert self.image_d is not None and self.noise_d_low is not None
            self.log_snr = partial(
                _log_snr_schedule_cosine_shifted,
                image_d=self.image_d,
                noise_d=self.noise_d_low,
            )
        elif self.noise_schedule == "cosine_interpolated":
            assert (
                self.image_d is not None
                and self.noise_d_low is not None
                and self.noise_d_high is not None
            )
            self.log_snr = partial(
                _log_snr_schedule_cosine_interpolated,
                image_d=self.image_d,
                noise_d_low=self.noise_d_low,
                noise_d_high=self.noise_d_high,
            )
        else:
            raise ValueError(f"invalid beta schedule: {self.noise_schedule}")

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # continuous timesteps
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def get_network_condition(self, steps=None, input_dict=None, only_custom_condition=False):
        if only_custom_condition:
            other_condition = self.condition_model(input_dict)
            return dict(other_condition=other_condition)
        time_condition = self.log_snr(steps)[:, 0, 0, 0]
        other_condition = self.condition_model(input_dict)
        return dict(time_condition=time_condition, other_condition=other_condition)

    def get_target(self, x_0, step_t, noise):
        if self.objective == "eps":
            target = noise
        elif self.objective == "x_0":
            target = x_0
        elif self.objective == "v":
            log_snr = self.log_snr(step_t)
            alpha, sigma = _log_snr_to_alpha_sigma(log_snr)
            target = alpha * noise - sigma * x_0
        else:
            raise ValueError(f"invalid objective {self.objective}")
        return target

    def get_loss_weight(self, steps):
        log_snr = self.log_snr(steps)
        snr = log_snr.exp()
        clipped_snr = snr.clone()
        if self.min_snr_loss_weight:
            clipped_snr.clamp_(max=self.min_snr_gamma)
        if self.objective == "eps":
            loss_weight = clipped_snr / snr
        elif self.objective == "x_0":
            loss_weight = clipped_snr
        elif self.objective == "v":
            loss_weight = clipped_snr / (snr + 1)
        else:
            raise ValueError(f"invalid objective {self.objective}")
        return loss_weight

    @autocast(enabled=False)
    def q_step_from_x_0(self, x_0, step_t, rng=None):
        # forward diffusion process q(zt|x0) where 0<t<1
        noise = self.randn_like(x_0, rng=rng)
        log_snr = self.log_snr(step_t)
        alpha, sigma = _log_snr_to_alpha_sigma(log_snr)
        x_t = x_0 * alpha + noise * sigma
        return x_t, noise

    def q_step(self, x_s, step_t, step_s, rng=None):
        # q(zt|zs) where 0<s<t<1
        # cf. Appendix A of https://arxiv.org/pdf/2107.00630.pdf
        log_snr_t = self.log_snr(step_t)
        log_snr_s = self.log_snr(step_s)
        alpha_t, sigma_t = _log_snr_to_alpha_sigma(log_snr_t)
        alpha_s, sigma_s = _log_snr_to_alpha_sigma(log_snr_s)
        alpha_ts = alpha_t / alpha_s
        var_noise = self.randn_like(x_s, rng=rng)
        mean = x_s * alpha_ts
        var = sigma_t.pow(2) - alpha_ts.pow(2) * sigma_s.pow(2)
        x_t = mean + var.sqrt() * var_noise
        return x_t

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
        condition_dict.update(dict(time_condition=log_snr_t[:, 0, 0, 0]))
        if self.cond_mode == "concat" and isinstance(condition_dict['other_condition'], torch.Tensor):
                condition_feat = condition_dict['other_condition']
                time_condition_dict = dict(time_condition=condition_dict['time_condition'])
                prediction = self.model(torch.cat([x_t, condition_feat], dim=1), time_condition_dict)
        else:
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

    @torch.inference_mode()
    def sample(
        self,
        batch_dict: dict,
        batch_size: int,
        num_steps: int,
        progress: bool = True,
        rng: list[torch.Generator] | torch.Generator | None = None,
        return_all: bool = False,
        mode: Literal["ddpm", "ddim"] = "ddpm",
        ddim_eta: float = 0.0,
    ):
        x = self.randn(batch_size, *self.sampling_shape, rng=rng, device=self.device)
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

    @torch.inference_mode()
    def inpaint(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        batch_dict: dict,
        num_steps: int,
        num_resample_steps: int = 1,  # "n" of the RePaint paper
        jump_length: int = 1,
        progress: bool = True,
        rng: list[torch.Generator] | torch.Generator | None = None,
        return_all: bool = False,
    ):
        assert num_resample_steps > 0
        assert jump_length > 0
        batch_size = known.shape[0]
        x_t = self.randn(batch_size, *self.sampling_shape, rng=rng, device=self.device)
        condition_dict = self.get_network_condition(input_dict=batch_dict, only_custom_condition=True)
        steps = torch.linspace(1, 0, num_steps + 1, device=self.device)
        steps = steps[None].repeat_interleave(batch_size, dim=0)

        if return_all:
            out = [x_t]

        for i in tqdm(
            range(num_steps), desc="RePaint", leave=False, disable=not progress
        ):
            for j in range(num_resample_steps):
                step_t = steps[:, [i]]
                step_s = steps[:, [i + 1]]
                interp = torch.linspace(0, 1, jump_length + 1, device=self.device)
                r_steps = step_t + interp[None] * (step_s - step_t)

                # t->s (reverse diffusion)
                x = x_t
                for k in range(jump_length):
                    r_step_t = r_steps[:, k]
                    r_step_s = r_steps[:, k + 1]
                    known_s, _ = self.q_step_from_x_0(known, r_step_s, rng=rng)
                    unknown_s = self.p_step(x, condition_dict, r_step_t, r_step_s, rng=rng)
                    x = mask * known_s + (1 - mask) * unknown_s
                x_s = x

                if return_all:
                    out.append(x_s)

                if (i == num_steps - 1) or (j == num_resample_steps - 1):
                    x_t = x
                    break

                # s->t (forward diffusion)
                x = x_s
                for k in range(jump_length, 0, -1):
                    r_step_t = r_steps[:, k - 1]
                    r_step_s = r_steps[:, k]
                    x = self.q_step(x, condition_dict, r_step_t, r_step_s, rng=rng)
                x_t = x


        # steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        # steps = steps[None].repeat_interleave(batch_size, dim=0)
        # p_step_kwargs = dict(rng=rng, mode='ddpm', ddim_eta=0.0)
        # tqdm_kwargs = dict(desc="sampling", leave=False, disable=not progress)
        # for i in tqdm(range(num_steps), **tqdm_kwargs):
        #     step_t = steps[:, i]
        #     step_s = steps[:, i + 1]
        #     x = self.p_step(x, condition_dict, step_t, step_s, **p_step_kwargs)
        #     if return_all:
        #         out.append(x)

        return torch.stack(out) if return_all else x_s

    @torch.inference_mode()
    def repaint(
        self,
        known: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int,
        num_resample_steps: int = 1,  # "n" of the RePaint paper
        jump_length: int = 1,
        progress: bool = True,
        rng: list[torch.Generator] | torch.Generator | None = None,
        return_all: bool = False,
    ):
        # re-implementation of RePaint (https://arxiv.org/abs/2201.09865)
        assert num_resample_steps > 0
        assert jump_length > 0
        batch_size = known.shape[0]
        x_t = self.randn(batch_size, *self.sampling_shape, rng=rng, device=self.device)
        steps = torch.linspace(1, 0, num_steps + 1, device=self.device)
        steps = steps[None].repeat_interleave(batch_size, dim=0)

        if return_all:
            out = [x_t]

        for i in tqdm(
            range(num_steps), desc="RePaint", leave=False, disable=not progress
        ):
            for j in range(num_resample_steps):
                step_t = steps[:, [i]]
                step_s = steps[:, [i + 1]]
                interp = torch.linspace(0, 1, jump_length + 1, device=self.device)
                r_steps = step_t + interp[None] * (step_s - step_t)

                # t->s (reverse diffusion)
                x = x_t
                for k in range(jump_length):
                    r_step_t = r_steps[:, k]
                    r_step_s = r_steps[:, k + 1]
                    known_s, _ = self.q_step_from_x_0(known, r_step_s, rng=rng)
                    unknown_s = self.p_step(x, r_step_t, r_step_s, rng=rng)
                    x = mask * known_s + (1 - mask) * unknown_s
                x_s = x

                if return_all:
                    out.append(x_s)

                if (i == num_steps - 1) or (j == num_resample_steps - 1):
                    x_t = x
                    break

                # s->t (forward diffusion)
                x = x_s
                for k in range(jump_length, 0, -1):
                    r_step_t = r_steps[:, k - 1]
                    r_step_s = r_steps[:, k]
                    x = self.q_step(x, r_step_t, r_step_s, rng=rng)
                x_t = x

        return torch.stack(out) if return_all else x_s

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
        if self.cond_mode == "concat":
            if isinstance(condition['other_condition'], torch.Tensor):
                condition_feat = condition.pop('other_condition')
                x_t = torch.cat([x_t, condition_feat], dim=1)
                
        prediction = self.model(x_t, condition)
        target = self.get_target(x_0, steps, noise)
        loss = self.criterion(prediction, target)  # (B,C,H,W)
        loss = einops.reduce(loss * loss_mask, "B ... -> B ()", "sum")
        loss_mask = einops.reduce(loss_mask, "B ... -> B ()", "sum")
        loss = loss / loss_mask.add(1e-8)  # (B,)
        loss = (loss * self.get_loss_weight(steps)).mean()
        return loss

    def forward(
        self,
        input_dict: dict,
    ) -> torch.Tensor:
        # shared in continuous/discrete versions
        x_0 = input_dict['x_0']
        steps = self.sample_timesteps(x_0.shape[0], x_0.device)
        if self.w_loss_weight:
            B, C, _, _ = x_0.shape
            loss_mask = input_dict.get('scene_loss_weight_map', None) # B H W
            if loss_mask is not None:
                # expand to B C H W
                loss_mask = loss_mask.unsqueeze(1).repeat(1, C, 1, 1)

        else:
            loss_mask = None
        loss = self.p_loss(input_dict, steps, loss_mask)
        return loss