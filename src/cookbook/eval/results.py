from cookbook.eval.datalake import FindExperiments, MetricsAll
from cookbook.constants import ALL_NAMED_GROUPS

from .miniframe import MiniFrame

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
) -> tuple[MiniFrame, MiniFrame]:
    experiments = FindExperiments.run(dashboard=dashboard)
    metrics = MetricsAll.prun(experiment_id=[experiment.experiment_id for experiment in experiments])

    all_metrics_table = MiniFrame(title=dashboard)

    for metric in metrics:
        if metric.is_aggregate:
            # we skip aggregate tasks; we will aggregate them ourselves...
            continue

        # we add primary metric after checking that we have a model name
        assert metric.model_name is not None
        assert metric.alias is not None

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
        all_metrics_table.add(col=metric.alias, row=metric.model_name, val=metric.metrics.primary_score)

        # add bpb if available and selected
        if metric.metrics.bpb is not None and show_bpb:
            all_metrics_table.add(col=f"{metric.alias}:bpb", row=metric.model_name, val=metric.metrics.bpb)

    # remove metrics that do not have any models evaluated against them
    if not show_partial:
        all_metrics_table = all_metrics_table.drop_empty()

    avg_metrics_table = MiniFrame(title=dashboard)

    for group_name, tasks in ALL_NAMED_GROUPS.items():
        if "mmlu" in group_name and not average_mmlu:
            continue
        elif "core" in group_name and not average_core:
            continue
        elif "gen" in group_name and not average_generative:
            continue

        tasks_table = all_metrics_table.keep_cols(tasks)
        if len(tasks_table) == 0:
            # no need to keep averages for groups that have no models evaluated against their tasks
            continue

        for model, scores in tasks_table.rows:
            average = sum(filtered_scores := [s for s in scores if s is not None]) / len(filtered_scores)
            avg_metrics_table.add(col=group_name, row=model, val=average)

    return all_metrics_table, avg_metrics_table



if __name__ == "__main__":
    import sys
    all_metrics, all_averages = make_dashboard_table(sys.argv[1])


    all_averages.sort(col="mmlu:mc", reverse=True).show()
