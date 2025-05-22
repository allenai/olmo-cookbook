import logging
import re
from dataclasses import dataclass, field

from cookbook.constants import ALL_NAMED_GROUPS
from cookbook.eval.datalake import FindExperiments, MetricsAll
from cookbook.eval.miniframe import MiniFrame


logger = logging.getLogger(__name__)


@dataclass
class DashboardTables:
    metrics: MiniFrame
    averages: MiniFrame
    missing_tasks: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_title(cls, title: str) -> "DashboardTables":
        return cls(metrics=MiniFrame(title=title), averages=MiniFrame(title=title))

    def find_missing_tasks(self, metrics: list[MetricsAll]):
        """
        This method looks for missing task-model combinations. A missing task for a model is a task whose
        result exists only for some models.

        We need to keep track of these missing tasks because they will skew averages in unexpected ways
        otherwise.

        They are also useful for users to debug issues with Datalake, since they can identify which Beaker
        runs have not been uploaded correctly.
        """
        # Get unique model and task names
        unique_model_names = {metric.model_name for metric in metrics if metric.model_name is not None}
        unique_task_names = {metric.alias for metric in metrics if metric.alias is not None}

        assert len(self.missing_tasks) == 0, "Missing tasks already found"

        # Check for missing task-model combinations
        for model in unique_model_names:
            missing_tasks: list[str] = []
            for task in unique_task_names:
                if not any(m.model_name == model and m.alias == task for m in metrics):
                    missing_tasks.append(task)

            # only append a model if it has 1+ missing task
            if missing_tasks:
                self.missing_tasks[model] = missing_tasks


def make_dashboard_table(
    dashboard: str,
    task_regex: str = r"^.*$",
    model_regex: str = r"^.*$",
    average_mmlu: bool = True,
    average_core: bool = True,
    average_generative: bool = True,
    show_bpb: bool = False,
    show_mc: bool = True,
    show_rc: bool = False,
    show_generative: bool = True,
    show_partial: bool = True,
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

    # validate that all models have completed all tasks
    tables.find_missing_tasks(metrics)

    for metric in metrics:
        if metric.is_aggregate:
            # we skip aggregate tasks; we will aggregate them ourselves...
            continue

        # we add primary metric after checking that we have a model name
        assert metric.model_name is not None

        if not re.search(task_regex, metric.alias):
            # skip if task does not match regex
            continue

        if not re.search(model_regex, metric.model_name):
            # skip if model does not match regex
            continue

        if ":mc:" in metric.alias and not show_mc:
            # skip if it is a mc metric and MC is not selected
            continue
        elif ":rc:" in metric.alias and not show_rc:
            # skip if it is a rc metric and RC is not selected
            continue
        elif ":bpb:" in metric.alias and not show_bpb:
            # skip if it is a bpb metric and BPB is not selected
            continue
        elif not show_generative:
            # skip if it is a generative metric and GENERATIVE is not selected
            continue

        # add primary score
        tables.metrics.add(col=metric.alias, row=metric.model_name, val=metric.metrics.primary_score)

        # add bpb if available and selected
        if metric.metrics.bpb is not None and show_bpb:
            tables.metrics.add(col=f"{metric.alias}:bpb", row=metric.model_name, val=metric.metrics.bpb)

    # remove metrics that do not have any models evaluated against them
    if not show_partial:
        tables.metrics = tables.metrics.drop_empty()

    for group_name, tasks in ALL_NAMED_GROUPS.items():
        if "mmlu" in group_name and not average_mmlu:
            continue
        elif "core" in group_name and not average_core:
            continue
        elif "gen" in group_name and not average_generative:
            continue

        tasks_table = tables.metrics.keep_cols(*tasks)
        if len(tasks_table) == 0:
            # no need to keep averages for groups that have no models evaluated against their tasks
            continue

        for model, scores in tasks_table.rows:
            filtered_scores = [s for s in scores if s is not None]
            average = (sum(filtered_scores) / len(filtered_scores)) if filtered_scores else 0.0
            tables.averages.add(col=group_name, row=model, val=average)

    return tables
