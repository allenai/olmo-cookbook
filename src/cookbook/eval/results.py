from datetime import datetime
import logging
import re
from typing import NamedTuple

from cookbook.constants import ALL_NAMED_GROUPS
from cookbook.eval.datalake import FindExperiments, MetricsAll
from cookbook.eval.miniframe import MiniFrame

logger = logging.getLogger(__name__)

RE_MC_TASK = re.compile(r"(?P<prefix>:mc)($|:)")
RE_RC_TASK = re.compile(r"(?P<prefix>:rc)($|:)")
RE_SUITE_TASK = re.compile(r"(?P<prefix>::.+)$")


class DashboardTables(NamedTuple):
    metrics: MiniFrame
    averages: MiniFrame
    missing_tasks: dict[str, list[str]]

    @classmethod
    def from_title(cls, title: str) -> "DashboardTables":
        return cls(metrics=MiniFrame(title=title), averages=MiniFrame(title=title), missing_tasks={})


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
) -> DashboardTables:
    experiments = FindExperiments.run(dashboard=dashboard)

    logger.info(f"Found {len(experiments)} experiments in dashboard {dashboard}")

    # these are the tables that will be displayed on the dashboard
    tables = DashboardTables.from_title(dashboard)

    if len(experiments) == 0:
        # return empty tables if no experiments are found
        return tables

    metrics = MetricsAll.prun(
        experiment_id=[experiment.experiment_id for experiment in experiments],
        force=[force for _ in experiments],
        skip_on_fail=[skip_on_fail for _ in experiments],
    )

    # keep track of bpb metrics names; we need these to warn users if a metric is missing,
    # but we wanna report the original metric name in the warning.
    bpb_to_og_metric_name_map: dict[str, str] = {}
    
    # Filter to keep only the newest metric for each (model_name, alias) pair
    unique_metrics = {}
    for metric in metrics:
        key = (metric.model_name, metric.alias)
        if key in unique_metrics:
            metric_dt = datetime.fromisoformat(metric.current_date.replace(" UTC", "+00:00"))
            existing_dt = datetime.fromisoformat(unique_metrics[key].current_date.replace(" UTC", "+00:00"))
            if metric_dt > existing_dt:
                unique_metrics[key] = metric
        else:
            unique_metrics[key] = metric
    metrics = list(unique_metrics.values())

    for metric in metrics:
        if metric.is_aggregate:
            # we skip aggregate tasks; we will aggregate them ourselves...
            continue

        # we add primary metric after checking that we have a model name
        assert metric.model_name is not None

        # add primary score
        tables.metrics.add(col=metric.alias, row=metric.model_name, val=metric.metrics.primary_score)

        # add bpb if available and selected
        if metric.metrics.bpb is not None:
            if (bpb_alias := make_bpb_name(metric.alias)) is not None:
                tables.metrics.add(col=bpb_alias, row=metric.model_name, val=metric.metrics.bpb)
                bpb_to_og_metric_name_map[bpb_alias] = metric.alias

    for model_row in tables.metrics.rows:
        for metric_column_name, metric_column_value in zip(model_row.columns, model_row.values):
            # check if any of the values are None; if all values are there, this metric is ok,
            # we have all results!
            if metric_column_value is not None:
                continue

            # replace if necessary
            metric_column_name = bpb_to_og_metric_name_map.get(metric_column_name, metric_column_name)

            # add missing tasks to the missing_tasks dict
            tables.missing_tasks.setdefault(model_row.name, []).append(metric_column_name)

    # add groups for tasks that have a bpb group.
    # we add them here, instead of adding to ALL_NAMED_GROUPS directly, because these are not
    # valid task names in OE-Eval, thus we don't want ppl to launch experiments based on them.
    expanded_named_groups = {
        **ALL_NAMED_GROUPS,
        **{
            bpb_group: bpb_tasks
            for group, tasks in ALL_NAMED_GROUPS.items()
            if (bpb_group := make_bpb_name(group)) is not None
            and all(bpb_tasks := [make_bpb_name(task) for task in tasks])
            and len(bpb_tasks) > 0
        },
    }

    for group_name, tasks in expanded_named_groups.items():
        tasks_table = tables.metrics.keep_cols(*tasks)
        if len(tasks_table) == 0:
            # no need to keep averages for groups that have no models evaluated against their tasks
            continue

        for row in tasks_table.rows:
            filtered_scores = [s for s in row.values if s is not None]
            average = (sum(filtered_scores) / len(filtered_scores)) if filtered_scores else 0.0
            tables.averages.add(col=group_name, row=row.name, val=average)

    return tables
