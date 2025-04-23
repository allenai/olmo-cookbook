from enum import Enum
from typing import List, Literal, Union

from olmo_eval import list_tasks


DownstreamEvaluator = Enum("DownstreamEvaluator", {name: name for name in Union[list_tasks()] + ["ALL"]})


def get_all_tasks() -> List[str]:
    """Return all downstream evaluation task names except 'all'."""
    return [task.value for task in DownstreamEvaluator if task.name != "ALL"]  # type: ignore
