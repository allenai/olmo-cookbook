import pprint
from cookbook.constants import ALL_NAMED_GROUPS

from .datalake import get_simple_dashboard_metrics


def simple_dashboard(
    dashboard: str,
    tasks: list[str],
    models: list[str],
    cache_dir: str,
    invalidate_cache: bool,
    debug: bool,
):
    valid_tasks = set(task for task_group in tasks for task in ALL_NAMED_GROUPS.get(task_group, [task_group]))
    valid_models = set(models)
    metrics = get_simple_dashboard_metrics(
        dashboard_name=dashboard,
        cache_dir=cache_dir,
        invalidate_cache=invalidate_cache,
        debug=debug,
    )

    to_print: dict[str, dict[str, float]] = {}
    for metric in metrics:
        if len(valid_tasks) > 0 and metric['task_name'] not in valid_tasks:
            continue
        if len(valid_models) > 0 and metric['model_name'] not in valid_models:
            continue

        score = round(metric['metrics']['primary_score'], 4)
        to_print.setdefault(metric['task_name'], {})[metric['model_name']] = score

    pprint.pprint(to_print)
