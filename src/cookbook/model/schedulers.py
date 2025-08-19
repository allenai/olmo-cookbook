import logging
import warnings
from dataclasses import dataclass
from math import pi, cos
from typing import Optional, Union

import torch
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import Scheduler

log = logging.getLogger(__name__)


@dataclass
class HalfCosWithWarmup(Scheduler):
    """
    Second half of a cosine learning rate schedule, with a warmup before that.
    Note: This assumes that the peak LR set is for the full cosine schedule.
    """

    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    alpha_f: float = 0.1
    t_max: Optional[int] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self):
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (self.warmup_fraction < 0 or self.warmup_fraction > 1):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        t_max = t_max if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current < warmup:
            max_lr = eta_min + (initial_lr - eta_min) / 2
            return _linear_warmup(max_lr, current, warmup, self.warmup_min_lr)
        elif current >= t_max:
            return eta_min
        else:
            current = current - warmup
            t_max = t_max - warmup
            current += t_max
            t_max *= 2
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * current / t_max)) / 2


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
