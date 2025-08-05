from typing import List, Literal

import einops
import torch
from torch import nn
from torch.cuda.amp import autocast


class GaussianDiffusion(nn.Module):
    """
    Base class for continuous/discrete Gaussian diffusion models
    """

    def __init__(
        self,
        model: nn.Module,
        condition_model: nn.Module=None,
        sampling: Literal["ddpm", "ddim"] = "ddpm",
        prediction_type: Literal["eps", "v", "x_0"] = "eps",
        loss_type: Literal["l2", "l1", "huber"] | nn.Module = "l2",
        num_training_steps: int = 1000,
        noise_schedule: Literal["linear", "cosine", "sigmoid"] = "linear",
        min_snr_loss_weight: bool = True,
        min_snr_gamma: float = 5.0,
        sampling_resolution: tuple[int, int] | None = None,
        clip_sample: bool = True,
        clip_sample_range: float = 1,
    ):
        super().__init__()
        self.model = model
        self.condition_model = condition_model
        self.sampling = sampling
        self.num_training_steps = num_training_steps
        self.objective = prediction_type
        self.noise_schedule = noise_schedule
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        if loss_type == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        elif loss_type == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif isinstance(loss_type, nn.Module):
            self.criterion = loss_type
        else:
            raise ValueError(f"invalid criterion: {loss_type}")
        if hasattr(self.criterion, "reduction"):
            assert self.criterion.reduction == "none"

        if sampling_resolution is None:
            assert hasattr(self.model, "resolution")
            assert hasattr(self.model, "in_channels")
            self.sampling_shape = (
                self.model.in_channels,
                *self.model.resolution,
            )
        else:
            assert len(sampling_resolution) == 2
            assert hasattr(self.model, "in_channels")
            self.sampling_shape = (self.model.in_channels, *sampling_resolution)

        self.setup_parameters()
        self.register_buffer("_dummy", torch.tensor([]))

    @property
    def device(self):
        return self._dummy.device

    def randn(
        self,
        *shape,
        rng: List[torch.Generator] | torch.Generator | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if rng is None:
            return torch.randn(*shape, **kwargs)
        elif isinstance(rng, torch.Generator):
            return torch.randn(*shape, generator=rng, **kwargs)
        elif isinstance(rng, list):
            assert len(rng) == shape[0]
            return torch.stack(
                [torch.randn(*shape[1:], generator=r, **kwargs) for r in rng]
            )
        else:
            raise ValueError(f"invalid rng: {rng}")

    def randn_like(
        self,
        x: torch.Tensor,
        rng: List[torch.Generator] | torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.randn(*x.shape, rng=rng, device=x.device, dtype=x.dtype)

    def setup_parameters(self) -> None:
        raise NotImplementedError

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def get_network_condition(self, steps: torch.Tensor):
        raise NotImplementedError

    def get_target(self, x_0, steps, noise):
        raise NotImplementedError

    def get_loss_weight(self, steps):
        raise NotImplementedError

    @autocast(enabled=False)
    def q_step_from_x_0(self, x_0, steps, rng):
        raise NotImplementedError

    def q_step(self, *args, **kwargs):
        raise NotImplementedError

    @torch.inference_mode()
    def p_step(self, *args, **kwargs):
        raise NotImplementedError

    def p_loss(
        self,
        x_0: torch.Tensor,
        steps: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # shared in continuous/discrete versions
        loss_mask = torch.ones_like(x_0) if loss_mask is None else loss_mask
        x_t, noise = self.q_step_from_x_0(x_0, steps)
        condition = self.get_network_condition(steps)
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
        x_0: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # shared in continuous/discrete versions
        steps = self.sample_timesteps(x_0.shape[0], x_0.device)
        loss = self.p_loss(x_0, steps, loss_mask)
        return loss

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        num_steps: int,
        progress: bool,
        rng: list[torch.Generator] | torch.Generator | None,
        return_all: bool,
        mode: str,
        *args,
        **kwargs,
    ):
        raise NotImplementedError
