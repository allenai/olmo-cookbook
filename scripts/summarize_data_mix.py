"""

Examples:
    Peteish7    python scripts/summarize_data_mix.py https://wandb.ai/ai2-llm/olmo-medium/runs/cej4ya39
    OLMoE       python scripts/summarize_data_mix.py https://wandb.ai/ai2-llm/olmoe/runs/rzsn9tlc
    Amberish    python scripts/summarize_data_mix.py https://wandb.ai/ai2-llm/olmo-medium/runs/ij4ls6v2

"""

import logging
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple, Union

import click

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")  # Simple formatter to just show the message
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


run_path_re = re.compile(r"^[^/]+/[^/]+/[^/]+$")
run_path_url = re.compile(r"^https?://wandb.ai/([^/]+)/([^/]+)/runs/([^/]+)")


def flatten_dict(dictionary, parent_key="", separator=".", include_lists=False):
    """
    Flatten a nested dictionary into a single-level dictionary.

    Args:
        dictionary (dict): The nested dictionary to be flattened.
        parent_key (str, optional): The parent key to be prepended to the keys of the flattened dictionary. Defaults to "".
        separator (str, optional): The separator to be used between the parent key and the keys of the flattened dictionary. Defaults to ".".
        include_lists (bool, optional): Whether to convert lists to dictionaries with integer keys. Defaults to False.

    Returns:
        dict: The flattened dictionary.

    """
    d: Dict[str, Any] = {}
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        # convert lists to dict with key <int>
        if isinstance(value, list) and include_lists:
            value = {f"{i}": v for i, v in enumerate(value)}
        if isinstance(value, MutableMapping):
            d.update(
                **flatten_dict(value, new_key, separator=separator, include_lists=include_lists)
            )
        else:
            d[new_key] = value
    return d


def parse_run_path(run_path: str) -> str:
    """For convenience, we allow run paths as well as URLs."""
    run_path = run_path.strip("/")
    if run_path_re.match(run_path):
        return run_path

    m = run_path_url.match(run_path)
    if m is not None:
        entity, project, run_id = m.groups()
        return f"{entity}/{project}/{run_id}"

    raise ValueError(f"Could not parse '{run_path}'")


def format_counter_paths(data_paths, log):
    """
    Format a Counter containing file paths into a readable log output.
    Shows full paths with aligned counts and percentages.

    Args:
        counter (Counter): Counter object containing path counts
        log (logging.Logger): Logger instance to output the formatted results
    """
    if not data_paths:
        log.info("Counter is empty")
        return

    # Find the largest count for padding
    max_count_width = len(str(max(data_paths.values())))

    # Sort by count in descending order
    sorted_items = data_paths.most_common()
    total_count = sum(data_paths.values())

    log.info(f"Total entries: {total_count}")
    log.info("-" * 120)  # Made longer to accommodate full paths

    for path, count in sorted_items:
        # Format the percentage
        percentage = (count / total_count) * 100

        # Create the formatted string with aligned counts and percentages
        formatted_line = f"{count:>{max_count_width},d} items ({percentage:5.1f}%) | {path}"

        log.info(formatted_line)

    log.info("-" * 120)  # Made longer to accommodate full paths


@click.command()
@click.argument(
    "run_path",
    type=str,
)
def main(run_path: str):
    import wandb

    api = wandb.Api()
    run = api.run(parse_run_path(run_path))

    config_raw = run._attrs["rawconfig"]

    # flattening the dict will make diffs easier
    config = flatten_dict(config_raw)

    # first, data.paths can be grouped and counted.
    data_paths = Counter([os.path.dirname(path) for path in config["data.paths"]])

    format_counter_paths(data_paths, log)


if __name__ == "__main__":
    main()
