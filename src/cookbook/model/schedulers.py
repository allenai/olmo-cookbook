from typing import Optional, Union
from dataclasses import dataclass
from olmo_core.exceptions import OLMoConfigurationError
import torch

from olmo_core.optim import Scheduler


@dataclass
# NOTE: Temporary port from https://github.com/allenai/OLMo-core/blob/dirkg/DenseExperiments/src/olmo_core/optim/scheduler.py#L67 for debugging
class WSD(Scheduler):
    """
    Warmup-stable-decay scheduler
    """

    warmup_steps: Optional[int] = 2000
    warmup_fraction: Optional[float] = None
    decay_steps: Optional[int] = None
    decay_fraction: Optional[float] = 0.1
    warmup_min_lr: float = 0.0
    decay_min_lr: float = 0.0

    def __post_init__(self):
        if (self.warmup_fraction is None) == (self.warmup_steps is None):
            raise OLMoConfigurationError("Either warmup_fraction or warmup_steps must be specified.")
        if self.warmup_fraction is not None and (self.warmup_fraction < 0 or self.warmup_fraction > 1):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

        if (self.decay_fraction is None) == (self.decay_steps is None):
            raise OLMoConfigurationError("Either decay_fraction or decay_steps must be specified.")
        if self.decay_fraction is not None and (self.decay_fraction < 0 or self.decay_fraction > 1):
            raise OLMoConfigurationError("decay_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup_steps is None:
            warmup_steps = round(max_steps * self.warmup_fraction) if self.warmup_fraction is not None else 0
        else:
            warmup_steps = self.warmup_steps

        if step <= warmup_steps:
            return _linear_warmup(initial_lr, step, warmup_steps, self.warmup_min_lr)

        if self.decay_steps is None:
            decay_steps = round(max_steps * self.decay_fraction) if self.decay_fraction is not None else 0
        else:
            decay_steps = self.decay_steps

        if step >= max_steps - decay_steps:
            return _linear_decay(initial_lr, max_steps - step, decay_steps, self.decay_min_lr)

        del step, max_steps
        return initial_lr


def _linear_warmup(
    initial_lr: Union[float, torch.Tensor], step: int, warmup_steps: int, warmup_min_lr: float = 0.0
) -> Union[float, torch.Tensor]:
    if isinstance(initial_lr, float):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= warmup_min_lr < initial_lr
    return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, warmup_steps) / warmup_steps


def _linear_decay(
    initial_lr: Union[float, torch.Tensor], step_from_end: int, decay_steps: int, decay_min_lr: float = 0.0
) -> Union[float, torch.Tensor]:
    if isinstance(initial_lr, float):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= decay_min_lr < initial_lr

    return decay_min_lr + (initial_lr - decay_min_lr) * min(step_from_end, decay_steps) / decay_steps
