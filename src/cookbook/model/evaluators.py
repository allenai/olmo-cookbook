from enum import Enum
from typing import List, Union

from olmo_eval import list_tasks


DownstreamEvaluator = Enum(
    "DownstreamEvaluator", {name.upper(): name for name in list_tasks() + ["all"]}, type=str
)


def get_all_tasks() -> List[str]:
    """Return all downstream evaluation tasks"""
    return [task.value for task in DownstreamEvaluator if task != DownstreamEvaluator.ALL]  # type: ignore
