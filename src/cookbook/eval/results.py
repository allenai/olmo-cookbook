import logging
import re
import sys

from cookbook.eval.datalake import FindExperiments, MetricsAll
from cookbook.eval.miniframe import MiniFrame

logger = logging.getLogger(__name__)

RE_MC_TASK = re.compile(r"(?P<prefix>:mc)($|:)")
RE_RC_TASK = re.compile(r"(?P<prefix>:rc)($|:)")
RE_SUITE_TASK = re.compile(r"(?P<prefix>::.+)$")


def make_bpb_name(alias: str) -> str | None:
    if ":bpb" in alias:
        # no double counting; this task already exists as bpb
        return None
    elif RE_MC_TASK.search(alias):
        # we dont do BPB on MC tasks
        return None
    elif RE_RC_TASK.search(alias):
        # we replace the :rc with :bpb
        return RE_RC_TASK.sub(":bpb\\2", alias)
    elif RE_SUITE_TASK.search(alias):
        return RE_SUITE_TASK.sub(":bpb\\1", alias)
    else:
        return f"{alias}:bpb"


def make_dashboard_table(
    dashboard: str,
    force: bool = False,
    skip_on_fail: bool = False,
) -> tuple[MiniFrame, dict[str, list[str]]]:
    experiments = FindExperiments.run(dashboard=dashboard)

    logger.info(f"Found {len(experiments)} experiments in dashboard {dashboard}")

    # # these are the tables that will be displayed on the dashboard
    # tables = DashboardTables.from_title(dashboard)

    metrics_table = MiniFrame(title=dashboard)
    missing_tasks: dict[str, list[str]] = {}

    if len(experiments) == 0:
        # return empty tables if no experiments are found
        return metrics_table, missing_tasks

    metrics = MetricsAll.prun(
        experiment_id=[experiment.experiment_id for experiment in experiments],
        force=[force for _ in experiments],
        skip_on_fail=[skip_on_fail for _ in experiments],
    )

    # keep track of bpb metrics names; we need these to warn users if a metric is missing,
    # but we wanna report the original metric name in the warning.
    bpb_to_og_metric_name_map: dict[str, str] = {}

    for metric in metrics:
        if metric.is_aggregate:
            # we skip aggregate tasks; we will aggregate them ourselves...
            continue

        # we add primary metric after checking that we have a model name
        assert metric.model_name is not None

        # add primary score
        metrics_table.add(col=metric.alias, row=metric.model_name, val=metric.metrics.primary_score)

        # add bpb if available and selected
        if metric.metrics.bpb is not None:
            if (bpb_alias := make_bpb_name(metric.alias)) is not None:
                metrics_table.add(col=bpb_alias, row=metric.model_name, val=metric.metrics.bpb)
                bpb_to_og_metric_name_map[bpb_alias] = metric.alias

    for model_row in metrics_table.rows:
        for metric_column_name, metric_column_value in zip(model_row.columns, model_row.values):
            # check if any of the values are None; if all values are there, this metric is ok,
            # we have all results!
            if metric_column_value is not None:
                continue

            # replace if necessary
            metric_column_name = bpb_to_og_metric_name_map.get(metric_column_name, metric_column_name)

            # add missing tasks to the missing_tasks dict
            missing_tasks.setdefault(model_row.name, []).append(metric_column_name)

    return metrics_table, missing_tasks


def print_missing_tasks(
    missing_tasks: dict[str, list[str]],
    rows_filter_models: list[str | re.Pattern],
    columns_filter_tasks: list[str | re.Pattern],
) -> None:

    # go through the missing models and tasks and print them out
    for model, missing_task in missing_tasks.items():
        # filter to only models user requested
        if rows_filter_models and not any(
            m.search(model) if isinstance(m, re.Pattern) else m == model for m in rows_filter_models
        ):
            continue

        # filter the missing tasks to only include the tasks that were requested
        model_missing_tasks = {
            task
            for task in missing_task
            if any(t.search(task) if isinstance(t, re.Pattern) else t == task for t in columns_filter_tasks)
        }
        if not model_missing_tasks:
            continue

        # the empty print adds a new line
        print(
            f"😱 Model \033[1m{model}\033[0m is missing \033[1m{len(model_missing_tasks)}\033[0m tasks:",
            file=sys.stderr,
        )
        for task in model_missing_tasks:
            print(f"  -{task}", file=sys.stderr)
        print(file=sys.stderr)
