from enum import Enum

from olmo_eval import list_tasks


class DownstreamEvaluator(Enum):
    """Enum class enumerating available in-loop evaluators."""

    ALL = "all"

    # Dynamically add tasks from olmo_eval.list_tasks()
    for task in list_tasks():
        locals()[task.upper()] = task
