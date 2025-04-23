from enum import Enum, auto
from typing import List, Literal, Union, cast, get_args

from olmo_eval import list_tasks


class DownstreamEvaluator(str, Enum):
    """Enum for downstream evaluation tasks."""

    ALL = "all"


for task_name in set(list_tasks()):
    task_upper = task_name.upper()
    if not hasattr(DownstreamEvaluator, task_upper):  # Avoid duplicates
        setattr(DownstreamEvaluator, task_upper, task_name)

EVALUATOR_NAMES = ("ALL",) + tuple(task.upper() for task in set(list_tasks()))
DownstreamEvaluatorType = Union[DownstreamEvaluator, Literal["ALL"]]


def get_all_tasks() -> List[str]:
    return [task for task in DownstreamEvaluator if task != DownstreamEvaluator.ALL]
