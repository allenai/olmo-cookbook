"""
Utility for visualizing and creating learning rate schedulers.
"""

import math
from typing import Optional, Dict, Tuple

from olmo_core.optim import SchedulerUnits
from cookbook.aliases import SchedulerConfig


class SchedulerPlot:
    """A callable class for learning rate scheduling based on OLMo-core scheduler implementations."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.scheduler_type = config.scheduler
        self.units = config.units

    def __call__(
        self,
        step: int,
        total_steps: int,
        warmup_steps: Optional[int] = None,
        base_lr: float = 1.0,
        min_lr: float = 0.0,
        **kwargs,
    ) -> float:
        if warmup_steps is None:
            if self.config.warmup is not None:
                warmup_steps = self.config.warmup
            elif self.config.warmup_fraction is not None:
                warmup_steps = int(total_steps * self.config.warmup_fraction)
            else:
                warmup_steps = 0

        scheduler_map = {
            "COSINE_LINEAR": self._handle_cosine_linear,
            "WSD": self._handle_wsd,
            "COSINE": self.cosine_with_warmup,
            "LINEAR": self.linear_with_warmup,
        }

        scheduler_fn = scheduler_map.get(self.scheduler_type, self.constant_scheduler)
        return scheduler_fn(step, total_steps, warmup_steps, base_lr, min_lr, **kwargs)

    def _handle_cosine_linear(self, step, total_steps, warmup_steps, base_lr, min_lr, **kwargs):
        decay_start_fraction = 0.9
        if self.config.decay_fraction is not None:
            decay_start_fraction = self.config.decay_fraction
        elif self.config.decay is not None:
            decay_start_fraction = 1 - (self.config.decay / total_steps)

        return self.cosine_with_warmup_and_decay(
            step, total_steps, warmup_steps, base_lr, min_lr, decay_start_fraction=decay_start_fraction
        )

    def _handle_wsd(self, step, total_steps, warmup_steps, base_lr, min_lr, **kwargs):
        stable_steps = kwargs.get("stable_steps", total_steps // 2)
        decay_steps = kwargs.get("decay_steps", total_steps // 3)

        if self.config.decay is not None:
            decay_steps = self.config.decay
        elif self.config.decay_fraction is not None:
            decay_steps = int(total_steps * self.config.decay_fraction)

        return self.wsd_scheduler(step, total_steps, warmup_steps, stable_steps, decay_steps, base_lr, min_lr)

    @staticmethod
    def constant_scheduler(
        step: int, total_steps: int, warmup_steps: int = 0, base_lr: float = 1.0, **kwargs
    ) -> float:
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)
        return base_lr

    @staticmethod
    def linear_with_warmup(
        step: int, total_steps: int, warmup_steps: int = 0, base_lr: float = 1.0, min_lr: float = 0.0, **kwargs
    ) -> float:
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)

        return base_lr - (base_lr - min_lr) * min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))

    @staticmethod
    def invsqrt_with_warmup(
        step: int, total_steps: int, warmup_steps: int = 0, base_lr: float = 1.0, min_lr: float = 0.0, **kwargs
    ) -> float:
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)

        decay_factor = math.sqrt(warmup_steps) / math.sqrt(max(step, 1))
        return max(min_lr, base_lr * decay_factor)

    @staticmethod
    def cosine_with_warmup(
        step: int, total_steps: int, warmup_steps: int = 0, base_lr: float = 1.0, min_lr: float = 0.0, **kwargs
    ) -> float:
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)

        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay

    @staticmethod
    def cosine_with_warmup_and_decay(
        step: int,
        total_steps: int,
        warmup_steps: int = 0,
        base_lr: float = 1.0,
        min_lr: float = 0.0,
        decay_start_fraction: float = 0.9,
        **kwargs,
    ) -> float:
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)

        decay_start = int(total_steps * decay_start_fraction)

        if step < decay_start:
            progress = (step - warmup_steps) / max(1, decay_start - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr + (base_lr - min_lr) * cosine_decay

        cosine_end_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi))
        return cosine_end_lr - (cosine_end_lr - min_lr) * (step - decay_start) / max(1, total_steps - decay_start)

    @staticmethod
    def wsd_scheduler(
        step: int,
        total_steps: int,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        base_lr: float = 1.0,
        min_lr: float = 0.0,
        **kwargs,
    ) -> float:
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)

        if step < warmup_steps + stable_steps:
            return base_lr

        if step < warmup_steps + stable_steps + decay_steps:
            decay_progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
            return base_lr - (base_lr - min_lr) * decay_progress

        return min_lr

    def _get_scheduler_parameters(
        self, total_units: int, global_batch_size: Optional[int] = None, **kwargs
    ) -> Tuple[Dict[str, int], str, Optional[Dict[str, int]]]:
        warmup_units = kwargs.get("warmup_steps", 0)
        if not warmup_units and self.config.warmup is not None:
            warmup_units = self.config.warmup
        elif not warmup_units and self.config.warmup_fraction is not None:
            warmup_units = int(total_units * self.config.warmup_fraction)

        unit_type = "steps" if self.units == SchedulerUnits.steps else "tokens"
        params = {
            "warmup_end": warmup_units,
            "total": total_units,
        }

        if self.scheduler_type == "WSD":
            stable_units = kwargs.get("stable_steps", 0)
            decay_units = kwargs.get("decay_steps", 0)

            if self.config.decay is not None:
                decay_units = self.config.decay
                stable_units = max(0, total_units - warmup_units - decay_units)
            elif self.config.decay_fraction is not None:
                decay_units = int(total_units * self.config.decay_fraction)
                stable_units = max(0, total_units - warmup_units - decay_units)
            else:
                if not decay_units:
                    decay_units = total_units // 3
                if not stable_units:
                    stable_units = max(0, total_units - warmup_units - decay_units)

            params["stable_end"] = warmup_units + stable_units
            params["decay_end"] = warmup_units + stable_units + decay_units

        if self.scheduler_type == "COSINE_LINEAR":
            decay_start_fraction = kwargs.get("decay_start_fraction", 0.9)

            if self.config.decay_fraction is not None:
                decay_start_fraction = self.config.decay_fraction
            elif self.config.decay is not None:
                decay_start_fraction = 1 - (self.config.decay / total_units)

            decay_start_units = int(total_units * decay_start_fraction)
            params["decay_start"] = decay_start_units

        step_equivalents = None
        if self.units == SchedulerUnits.tokens and global_batch_size is not None and global_batch_size > 0:
            step_equivalents = {key: value // global_batch_size for key, value in params.items()}

        return params, unit_type, step_equivalents

    def visualize(
        self,
        total_steps: int,
        base_lr: float = 1.0,
        min_lr: float = 0.0,
        width: int = 80,
        height: int = 20,
        samples_per_point: int = 10,
        show_annotations: bool = True,
        global_batch_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        params, unit_type, step_equivalents = self._get_scheduler_parameters(
            total_steps, global_batch_size, **kwargs
        )

        warmup_steps = params["warmup_end"]

        wsd_kwargs = {}
        if self.scheduler_type == "WSD" and "stable_end" in params:
            stable_steps = params["stable_end"] - params["warmup_end"]
            decay_steps = params["decay_end"] - params["stable_end"]
            wsd_kwargs = {"stable_steps": stable_steps, "decay_steps": decay_steps}
            kwargs.update(wsd_kwargs)

        high_res_steps = width * samples_per_point
        high_res_lrs = []

        for i in range(high_res_steps):
            step = int(i * total_steps / high_res_steps)
            high_res_lrs.append(self(step, total_steps, warmup_steps, base_lr, min_lr, **kwargs))

        lrs = []
        steps = []
        for i in range(width):
            start_idx = i * samples_per_point
            end_idx = start_idx + samples_per_point
            step = int(i * total_steps / width)
            steps.append(step)
            lrs.append(sum(high_res_lrs[start_idx:end_idx]) / samples_per_point)

        if max(lrs) == min(lrs):
            normalized_lrs = [height - 1] * len(lrs)
        else:
            normalized_lrs = [(height - 1) * (1 - (lr - min(lrs)) / (max(lrs) - min(lrs))) for lr in lrs]

        grid = [[" " for _ in range(width)] for _ in range(height)]

        y_axis_labels = []
        if height >= 10:
            lr_labels = [min(lrs) + i * (max(lrs) - min(lrs)) / 4 for i in range(5)]
            lr_labels.reverse()
            label_positions = [int(i * (height - 1) / 4) for i in range(5)]

            for i, pos in enumerate(label_positions):
                if pos < height:
                    label = f"{lr_labels[i]:.2e}" if lr_labels[i] < 0.001 else f"{lr_labels[i]:.5f}"
                    y_axis_labels.append((pos, label))

        for x in range(1, width - 1):
            y0 = normalized_lrs[x]
            y1 = normalized_lrs[x + 1]

            y0 = max(1, min(height - 2, y0))
            y1 = max(1, min(height - 2, y1))

            if abs(y1 - y0) < 1.0:
                y_int = int(y0)
                grid[y_int][x] = "█"
            else:
                if y0 > y1:
                    for y in range(int(y1), int(y0) + 1):
                        if 0 < y < height - 1:
                            char_idx = min(int(abs(y - int(y1)) / abs(int(y0) - int(y1)) * 3), 2)
                            grid[y][x] = ["▓", "▒", "░"][char_idx]
                else:
                    for y in range(int(y0), int(y1) + 1):
                        if 0 < y < height - 1:
                            char_idx = min(int(abs(y - int(y0)) / abs(int(y1) - int(y0)) * 3), 2)
                            grid[y][x] = ["░", "▒", "▓"][char_idx]

        transitions = {k: v for k, v in params.items() if k != "total"}

        if show_annotations and transitions:
            for name, step in transitions.items():
                x_pos = int(step * width / total_steps)
                if 0 < x_pos < width:
                    for y in range(height):
                        if grid[y][x_pos] == " ":
                            grid[y][x_pos] = "│"

        label_padding = 11

        print(f"\n{self.scheduler_type} Scheduler (max_lr={max(lrs):.2e}, min_lr={min(lrs):.2e}):")
        print(f"{' ' * label_padding}Learning Rate")
        print(f"{' ' * label_padding}┌" + "─" * width + "┐")

        for i, row in enumerate(grid):
            label = ""
            for pos, lbl in y_axis_labels:
                if pos == i:
                    label = lbl
                    break
            row_str = "".join(row[:width])
            padded_label = label.rjust(10) if label else " " * 10
            print(f"{padded_label} │{row_str}│")

        print(f"{' ' * label_padding}┴" + "─" * width + "┘")

        total_steps_formatted = f"{total_steps:,}".replace(",", "_")
        step_labels = f"0{' ' * (width - len(total_steps_formatted) - 1)}{total_steps_formatted}"
        print(f"{' ' * label_padding}{unit_type.capitalize()}: {step_labels}")

        if self.units == SchedulerUnits.tokens and step_equivalents:
            total_steps_equiv = step_equivalents["total"]
            total_steps_equiv_formatted = f"{total_steps_equiv:,}".replace(",", "_")
            steps_labels = f"0{' ' * (width - len(total_steps_equiv_formatted) - 1)}{total_steps_equiv_formatted}"
            print(f"{' ' * label_padding}Steps: {steps_labels}")

        if show_annotations and transitions:
            sorted_transitions = sorted(transitions.items(), key=lambda x: x[1])

            for name, step in sorted_transitions:
                x_pos = int(step * width / total_steps)
                if 0 <= x_pos < width:
                    step_formatted = f"{step:,}".replace(",", "_")

                    if self.units == SchedulerUnits.tokens and step_equivalents and name in step_equivalents:
                        steps_equiv = step_equivalents[name]
                        steps_equiv_formatted = f"{steps_equiv:,}".replace(",", "_")
                        label = f"{name}: {step_formatted} {unit_type} ({steps_equiv_formatted} steps)"
                    else:
                        label = f"{name}: {step_formatted} {unit_type}"

                    print(f"{' ' * label_padding}{label}")

        print("\n" + "=" * (width + label_padding) + "\n")


def test_scheduler():
    configs = [
        {"scheduler": "COSINE", "warmup": 1000, "units": SchedulerUnits.steps},
        {"scheduler": "LINEAR", "warmup_fraction": 0.1, "units": SchedulerUnits.steps},
        {"scheduler": "COSINE_LINEAR", "warmup": 1000, "decay_fraction": 0.8, "units": SchedulerUnits.steps},
        {"scheduler": "WSD", "warmup": 1000, "decay": 2000, "units": SchedulerUnits.steps},
        {"scheduler": "COSINE", "warmup": 1_000_000, "units": SchedulerUnits.tokens},
        {"scheduler": "LINEAR", "warmup_fraction": 0.1, "units": SchedulerUnits.tokens},
        {"scheduler": "COSINE_LINEAR", "warmup": 500_000, "decay_fraction": 0.8, "units": SchedulerUnits.tokens},
        {"scheduler": "WSD", "warmup": 1_000_000, "decay": 2_000_000, "units": SchedulerUnits.tokens},
    ]

    for i, config in enumerate(configs):
        scheduler = SchedulerPlot(SchedulerConfig(**config))
        if i < 4:
            scheduler.visualize(10000, 1e-3, 1e-5)
        else:
            scheduler.visualize(10_000_000, 1e-3, 1e-5, global_batch_size=2048)


if __name__ == "__main__":
    test_scheduler()
