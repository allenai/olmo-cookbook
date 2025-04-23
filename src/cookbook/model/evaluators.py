from enum import Enum

from olmo_eval import list_tasks


class DownstreamEvaluator(Enum):
    """Enum class enumerating available in-loop evaluators."""

    ALL = "all"

    # Dynamically add tasks from olmo_eval.list_tasks()
    for task in set(list_tasks()):
        task_upper = task.upper()
        if task_upper not in locals():
            locals()[task_upper] = task
