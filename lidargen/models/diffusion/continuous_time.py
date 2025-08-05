import math
from functools import partial
from typing import List, Literal

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.special import expm1
from tqdm.auto import tqdm

from . import base


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


class ContinuousTimeGaussianDiffusion(base.GaussianDiffusion):
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
    ):
        super().__init__(
            model=model,
            condition_model=condition_model,
            sampling="ddpm",
            prediction_type=prediction_type,
            loss_type=loss_type,
            num_training_steps=None,
            noise_schedule=noise_schedule,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma,
            sampling_resolution=sampling_resolution,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
        self.image_d = image_d
        self.noise_d_low = noise_d_low
        self.noise_d_high = noise_d_high

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

    def get_network_condition(self, steps):
        return self.log_snr(steps)[:, 0, 0, 0]

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
        prediction = self.model(x_t, log_snr_t[:, 0, 0, 0])
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
        batch_size: int,
        num_steps: int,
        progress: bool = True,
        rng: list[torch.Generator] | torch.Generator | None = None,
        return_all: bool = False,
        mode: Literal["ddpm", "ddim"] = "ddpm",
        ddim_eta: float = 0.0,
    ):
        x = self.randn(batch_size, *self.sampling_shape, rng=rng, device=self.device)
        if return_all:
            out = [x]
        steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        steps = steps[None].repeat_interleave(batch_size, dim=0)
        p_step_kwargs = dict(rng=rng, mode=mode, ddim_eta=ddim_eta)
        tqdm_kwargs = dict(desc="sampling", leave=False, disable=not progress)
        for i in tqdm(range(num_steps), **tqdm_kwargs):
            step_t = steps[:, i]
            step_s = steps[:, i + 1]
            x = self.p_step(x, step_t, step_s, **p_step_kwargs)
            if return_all:
                out.append(x)
        return torch.stack(out) if return_all else x

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
