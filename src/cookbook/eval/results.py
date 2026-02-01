from datetime import datetime
import logging
import re
import sys
from typing import NamedTuple

from cookbook.eval.datalake import FindExperiments, MetricsAll
from cookbook.eval.miniframe import MiniFrame
from cookbook.eval.named_tasks import (
    BaseNamedTasksGroup,
    NamedTasksGroupRegistry,
)

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
    skip_on_fail: bool = True,
) -> MiniFrame:
    experiments = FindExperiments.run(dashboard=dashboard)

    logger.info(f"Found {len(experiments)} experiments in dashboard {dashboard}")

    # # these are the tables that will be displayed on the dashboard
    # tables = DashboardTables.from_title(dashboard)

    metrics_table = MiniFrame(title=dashboard)

    if len(experiments) == 0:
        # return empty tables if no experiments are found
        return metrics_table

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
    metrics: list[MetricsAll] = list(unique_metrics.values())


    for metric in metrics:
        if metric.is_aggregate:
            # we skip aggregate tasks; we will aggregate them ourselves...
            continue

        # we add primary metric after checking that we have a model name
        assert metric.model_name is not None

        # @davidh: Hotfix for minerva math. The primary metric is set incorrectly in oe-eval but we
        # want to make 100% sure we're looking at the right metric, because a lot of midtraining eval
        # has already been ran. Fix here: https://github.com/allenai/oe-eval-internal/pull/571
        if 'minerva_math' in metric.alias and 'hamish_zs_reasoning' in metric.alias:
            metric.metrics.primary_score = metric.metrics.extra_metrics['exact_match_flex']

        # @davidh: Hotfix for Alpaca Eval tasks. The alpaca eval metric multiplies its score by 100. No PR
        # in oe-eval to avoid messing with adapt's backend.
        if 'alpaca' in metric.alias:
            metric.metrics.primary_score /= 100

        # @davidh: Hotfix for styled math. Fix here: https://github.com/allenai/oe-eval-internal/pull/592
        if 'styled_math500' in metric.alias and 'tulu' in metric.alias:
            metric.metrics.primary_score = metric.metrics.extra_metrics['exact_match_flex']

        # @soldni: Hotfix for wildchat/ultrachat masked ppl. The task has no alias in oe-eval,
        # so hash gets added at the end  the name. We strip if the hash matches the known one.
        if metric.alias in {'wildchat_masked_ppl-67b0e9', 'ultrachat_masked_ppl-831470'}:
            metric_alias, _ = metric.alias.rsplit('-', 1)
            metric.task_config.setdefault("metadata", {}).setdefault("alias", metric_alias)
            assert metric.alias in {'wildchat_masked_ppl', 'ultrachat_masked_ppl'}

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

    return metrics_table


class ExpandedTasks(NamedTuple):
    single_tasks: list[str | re.Pattern]
    named_groups: list[BaseNamedTasksGroup]
    all_column_tasks: list[str | re.Pattern]

    @classmethod
    def from_tasks(cls, tasks: list[str]) -> 'ExpandedTasks':
        # compile tasks names into regex patterns (if possible)
        compiled_tasks = [re.compile(task) if re.escape(task) != task else task for task in tasks]

        # we partition between single tasks and named groups; we also keep a set of all tasks names,
        # which we will use later to print any missing tasks.
        named_groups: list[BaseNamedTasksGroup] = []
        columns_filter_tasks: list[str | re.Pattern] = compiled_tasks[:]
        for compiled_task in compiled_tasks:
            matching_groups = [
                NamedTasksGroupRegistry.get(ng) for ng in NamedTasksGroupRegistry.search(compiled_task)
            ]
            named_groups.extend(matching_groups)
            columns_filter_tasks.extend(t for ng in matching_groups for t in ng.expanded_tasks)

        return cls(
            single_tasks=compiled_tasks,
            named_groups=named_groups,
            all_column_tasks=columns_filter_tasks,
        )


class ExpandedModels(NamedTuple):
    single_models: list[str | re.Pattern]

    @classmethod
    def from_models(cls, models: list[str]) -> 'ExpandedModels':
        # we filtered tasks, but the user might want to display only some models
        rows_filter_models: list[str | re.Pattern] = []
        if len(models) > 0:
            # Try to compile as regex only if it's clearly intended as a pattern (contains regex special chars)
            # BUT for exact model names (like from backfill), keep as string for exact matching
            # Check if the string looks like a regex pattern vs. a model name with special chars like '+'
            for m in models:
                # If it contains common model name patterns (step numbers with +), treat as literal string
                if '+' in m and 'step' in m:
                    # This looks like a model name with averaged steps, not a regex pattern
                    rows_filter_models.append(m)
                elif re.escape(m) != m:
                    # Contains regex special chars and doesn't look like a model name, compile as regex
                    rows_filter_models.append(re.compile(m))
                else:
                    # No special chars, keep as string for exact match
                    rows_filter_models.append(m)

        return cls(single_models=rows_filter_models)


def make_results_from_dashboard(
    dashboard_table: MiniFrame,
    tasks: list[str],
    models: list[str] | None = None
) -> MiniFrame:
    """
    Filter the results table based on the tasks and models provided.
    """
    expanded_tasks = ExpandedTasks.from_tasks(tasks)
    expanded_models = ExpandedModels.from_models(models or [])

    # start by filtering in all the single tasks
    results = dashboard_table.keep_cols(*expanded_tasks.single_tasks)

    # then iterate over named groups...
    for named_group in expanded_tasks.named_groups:
        # ...and try to combine them into a single score. Note we are giving it the full metrics table,
        # not the one after filtering to single tasks.
        combined_table = named_group.combine(dashboard_table)

        if combined_table is not None:
            # we manage to combine! lets put the combined score at the front
            results = combined_table + results
        else:
            # this cannot be combined. let's add each metric as a column. make sure not
            # to include duplicates.
            named_group_table = dashboard_table.keep_cols(*named_group.expanded_tasks)
            existing_columns = set(results.columns)
            named_group_table_only_new_columns = named_group_table.keep_cols(
                *(c for c in named_group_table.columns if c not in existing_columns)
            )

            # we add the new columns to the end of the table
            results = results + named_group_table_only_new_columns

    if expanded_models.single_models:
        results = results.keep_rows(*expanded_models.single_models)

    return results


def find_missing_tasks(results: MiniFrame, dashboard_table: MiniFrame | None = None) -> dict[str, list[str]]:
    """ Looks for columns that are set to None across all models (rows) in the results"""

    missing_tasks: dict[str, list[str]] = {}
    for model_row in results.rows:
        for metric_column_name, metric_column_value in zip(model_row.columns, model_row.values):
            # check if any of the values are None; if all values are there, this metric is ok,
            # we have all results!
            if metric_column_value is not None:
                continue

            all_tasks_set = set()
            try:
                # this is a task group! the get function will return a class that has an expanded_tasks attribute
                task_group = NamedTasksGroupRegistry.get(metric_column_name)

                # If we have access to the full dashboard table, check which individual tasks
                # in this group are actually missing instead of blindly expanding the entire group
                if dashboard_table is not None:
                    # Find the row for this model in the dashboard
                    dashboard_row = None
                    for row in dashboard_table.rows:
                        if row.name == model_row.name:
                            dashboard_row = row
                            break

                    if dashboard_row is not None:
                        # Check which individual tasks in the group are actually None
                        dashboard_cols = list(dashboard_row.columns)
                        dashboard_vals = list(dashboard_row.values)
                        for task in task_group.expanded_tasks:
                            task_str = str(task)  # Handle regex patterns
                            if task_str in dashboard_cols:
                                idx = dashboard_cols.index(task_str)
                                if dashboard_vals[idx] is None:
                                    all_tasks_set.add(task)
                            else:
                                # Task not in dashboard at all, so it's missing
                                all_tasks_set.add(task)
                    else:
                        # Model not found in dashboard, expand entire group
                        all_tasks_set.update(task_group.expanded_tasks)
                else:
                    # No dashboard table provided, fall back to expanding entire group
                    all_tasks_set.update(task_group.expanded_tasks)
            except ValueError:
                # actually not a task group, just a task name. append as is.
                all_tasks_set.add(metric_column_name)

            # add missing tasks to the missing_tasks dict
            missing_tasks.setdefault(model_row.name, []).extend(all_tasks_set)

    return missing_tasks


def print_missing_tasks(
    missing_tasks: dict[str, list[str]],
    tasks: list[str],
    models: list[str] | None = None,
) -> None:

    expanded_tasks = ExpandedTasks.from_tasks(tasks)
    expanded_models = ExpandedModels.from_models(models or [])

    # go through the missing models and tasks and print them out
    for model, missing_task in missing_tasks.items():
        # filter to only models user requested
        if expanded_models.single_models and not any(
            m.search(model) if isinstance(m, re.Pattern) else m == model for m in expanded_models.single_models
        ):
            continue

        # filter the missing tasks to only include the tasks that were requested
        model_missing_tasks = {
            task
            for task in missing_task
            if any(
                t.search(task) if isinstance(t, re.Pattern) else t == task
                for t in expanded_tasks.all_column_tasks
            )
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
