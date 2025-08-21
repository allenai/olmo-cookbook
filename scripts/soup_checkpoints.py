"""
Soup OLMo checkpoints.

Author: Luca Soldaini (@soldni)
Email:  lucas@allenai.org

"""


import argparse
import base64
from dataclasses import dataclass
import logging
from enum import Enum
import os
from pathlib import Path
import pickle
import re

from tqdm import tqdm


try:
    from safetensors import torch as st_torch
except ImportError as e:
    raise ImportError("Please install SafeTensors to use this script") from e


try:
    import torch
except ImportError as e:
    raise ImportError("Please install PyTorch to use this script") from e


@dataclass(eq=True, frozen=True)
class STKey:
    keys: tuple
    value_is_pickled: bool

    def encode(self) -> str:
        b = pickle.dumps((self.keys, self.value_is_pickled))
        b = base64.urlsafe_b64encode(b)
        return str(b, "ASCII")

    @classmethod
    def decode(cls, key: str) -> 'STKey':
        b = base64.urlsafe_b64decode(key)
        keys, value_is_pickled = pickle.loads(b)
        return cls(keys=keys, value_is_pickled=value_is_pickled)


def unflatten_dict(d: dict[STKey, torch.Tensor]) -> dict:
    result: dict = {}

    for key, value in d.items():
        if key.value_is_pickled:
            value = pickle.loads(value.numpy().data)

        target_dict = result
        for k in key.keys[:-1]:
            new_target_dict = target_dict.get(k)
            if new_target_dict is None:
                new_target_dict = {}
                target_dict[k] = new_target_dict
            target_dict = new_target_dict
        target_dict[key.keys[-1]] = value

    return result


def safetensors_file_to_state_dict(
    filename: Path | str,
    map_location: str | None = None
) -> dict[str, torch.Tensor]:

    if map_location is None:
        map_location = "cpu"
    state_dict = st_torch.load_file(filename, device=map_location)
    state_dict = {STKey.decode(k): v for k, v in state_dict.items()}
    return unflatten_dict(state_dict)


def get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class SoupType(Enum):
    uniform = "uniform"


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if path.exists() and path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)

    if (path / "model.pt").exists():
        return torch.load(path / "model.pt", map_location="cpu", weights_only=True)

    if (path / "model.safetensors").exists():
        return safetensors_file_to_state_dict(path / "model.safetensors")

    if (path / "model").exists() and (path / "model").is_dir():
        raise ValueError("Please unshard the checkpoint before souping.")

    raise FileNotFoundError(f"Could not find checkpoint in {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Soup OLMo checkpoints")
    parser.add_argument(
        "-c",
        "--checkpoints",
        type=Path,
        required=True,
        nargs="+",
        help="Path to checkpoint(s) to soup",
    )
    parser.add_argument(
        "-r",
        "--regex-layers",
        type=str,
        default=None,
        help="Regex to filter layers to soup; by default, all layers are souped. For layers that are not selected, weights from the first checkpoint are kept.",
    )
    parser.add_argument(
        "-s",
        "--soup-type",
        type=SoupType,
        default=SoupType.uniform,
        help=f"Methods for checkpoint souping. Choose from: {', '.join(SoupType.__members__.keys())}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to save the souped checkpoint",
    )
    opts = parser.parse_args()
    return opts


def main():
    logger = get_logger()
    args = parse_args()

    checkpoint_average: dict[str, torch.Tensor] = {}

    for i, path in enumerate(tqdm(args.checkpoints, desc="Loading checkpoints", position=0)):
        state_dict = load_checkpoint(path)

        if len(checkpoint_average) == 0:
            # initialize checkpoint_average with zeros
            checkpoint_average = {k: torch.zeros_like(v) for k, v in state_dict.items()}

        if any(k not in state_dict for k in checkpoint_average.keys()) or any(
            k not in checkpoint_average for k in state_dict.keys()
        ):
            raise ValueError(f"Checkpoint {path} has different keys")

        for k in tqdm(state_dict, desc="Summing checkpoints", position=1):
            if args.regex_layers and not re.search(args.regex_layers, k):
                if i != 0:
                    logger.info(f"Skipping {k} because it does not match {args.regex_layers}")
                else:
                    checkpoint_average[k] = state_dict[k]
                continue

            if state_dict[k].shape != checkpoint_average[k].shape:
                raise ValueError(f"Checkpoint {path} has different shape for key {k}")
            checkpoint_average[k] += state_dict[k] / len(args.checkpoints)

        # free memory
        del state_dict

    logger.info(f"Saving averaged checkpoint to {args.output}")
    # save the averaged checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_average, args.output / "model.pt")

    # copy configs, tokenizers, etc.
    for filename in os.listdir(args.checkpoints[0]):
        _, filename_extension = filename.split(".")
        if filename_extension not in {"json", "txt", "yaml", "md"}:
            continue
        logger.info(f"Copying {filename}")
        with open(args.checkpoints[0] / filename, "rb") as src_f, open(args.output / filename, "wb") as dst_f:
            dst_f.write(src_f.read())

    logger.info("Done!")


if __name__ == "__main__":
    main()
