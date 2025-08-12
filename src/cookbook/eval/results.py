from datetime import datetime
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


def make_pass_at_k_name(alias: str, k: int) -> str | None:
    return f"{alias}:pass_at_{k}"


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

    if len(experiments) == 0:
        # return empty tables if no experiments are found
        return metrics_table, {}

    metrics = MetricsAll.prun(
        experiment_id=[experiment.experiment_id for experiment in experiments],
        force=[force for _ in experiments],
        skip_on_fail=[skip_on_fail for _ in experiments],
    )

    # keep track of bpb metrics names; we need these to warn users if a metric is missing,
    # but we wanna report the original metric name in the warning.
    bpb_to_og_metric_name_map: dict[str, str] = {}

    # Filter to keep only the newest metric for each (model_name, alias) pair
    unique_metrics: dict[tuple[str | None, str], MetricsAll] = {}
    for metric in metrics:
        key = (metric.model_name, metric.alias)
        if key in unique_metrics:
            metric_dt = datetime.fromisoformat(metric.current_date.replace(" UTC", "+00:00"))
            existing_dt = datetime.fromisoformat(unique_metrics[key].current_date.replace(" UTC", "+00:00"))
            if metric_dt > existing_dt:
                unique_metrics[key] = metric
        else:
            unique_metrics[key] = metric
    filtered_metrics: list[MetricsAll] = list(unique_metrics.values())

    for metric in filtered_metrics:
        if metric.is_aggregate:
            # we skip aggregate tasks; we will aggregate them ourselves...
            continue

        # we add primary metric after checking that we have a model name
        assert metric.model_name is not None

        # @davidh: Hotfix for minerva math. The primary metric is set incorrectly in oe-eval but we
        # want to make 100% sure we're looking at the right metric, because a lot of midtraining eval
        # has already been ran. Fix here: https://github.com/allenai/oe-eval-internal/pull/571
        if "minerva_math" in metric.alias and "hamish_zs_reasoning" in metric.alias:
            metric.metrics.primary_score = metric.metrics.extra_metrics["exact_match_flex"]

        # @davidh: Hotfix for Alpaca Eval tasks. The alpaca eval metric multiplies its score by 100. No PR
        # in oe-eval to avoid messing with adapt's backend.
        if "alpaca" in metric.alias:
            metric.metrics.primary_score /= 100

        # @davidh: Hotfix for styled math. Fix here: https://github.com/allenai/oe-eval-internal/pull/592
        if "styled_math500" in metric.alias and "tulu" in metric.alias:
            metric.metrics.primary_score = metric.metrics.extra_metrics["exact_match_flex"]

        # add primary score
        metrics_table.add(col=metric.alias, row=metric.model_name, val=metric.metrics.primary_score)

        # add bpb if available and selected
        if metric.metrics.bpb is not None:
            if (bpb_alias := make_bpb_name(metric.alias)) is not None:
                metrics_table.add(col=bpb_alias, row=metric.model_name, val=metric.metrics.bpb)
                bpb_to_og_metric_name_map[bpb_alias] = metric.alias

        # add pass@4 if available and selected
        if metric.metrics.pass_at_4 is not None:
            if (pass_at_4_alias := make_pass_at_k_name(metric.alias, k=4)) is not None:
                metrics_table.add(col=pass_at_4_alias, row=metric.model_name, val=metric.metrics.pass_at_4)

        # add pass@16 if available and selected
        if metric.metrics.pass_at_16 is not None:
            if (pass_at_16_alias := make_pass_at_k_name(metric.alias, k=16)) is not None:
                metrics_table.add(col=pass_at_16_alias, row=metric.model_name, val=metric.metrics.pass_at_16)

    # Create missing tasks dictionary (currently empty as no tracking is implemented)
    missing_tasks: dict[str, list[str]] = {}

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
            print(f"  - {task}", file=sys.stderr)
        print(file=sys.stderr)
