from enum import Enum
from typing import List, Union

from olmo_eval import list_tasks


DownstreamEvaluator = Enum("DownstreamEvaluator", {name: name for name in Union[list_tasks()] + ["all"]})


def get_all_tasks() -> List[str]:
    """Return all downstream evaluation task names except 'all'."""
    return [task.value for task in DownstreamEvaluator if task.name != "all"]  # type: ignore
