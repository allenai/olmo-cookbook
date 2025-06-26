import logging
import os
import sys
import warnings
from typing import Optional

from cookbook.constants import ALL_NAMED_GROUPS
from cookbook.eval.results import make_bpb_name

sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from cookbook.analysis.stats import compute_significance

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_paired_comparison(
    df: pd.DataFrame,
    task: str,
    task_alias: list[str] | str,
    model_names: list[str],
    ax: Optional[plt.Axes],
    metric: str = "primary_score",
) -> dict:
    model_names = sorted(list(set(model_names)))

    _, p_values = compute_significance(
        df,
        models=model_names,
        metric=metric,
        alpha=0.05,  # significance level
        task=task,
        task_alias=task_alias,
        ax=ax,
        plot_sig_clusters=False,
    )

    # Return nothing if test failed to run
    if p_values is None:
        return {}

    mixes, scores, p_values, sig_clusters = p_values

    return {
        "mixes": mixes,
        "scores": scores,
        "p_values": p_values,
        "sig_clusters": sig_clusters,
        "task": task,
        "task_set": task,
        "metric": metric,
        "model_names": model_names,
    }


def run_instance_analysis(
    df, tasks, models, render_plots=True, plot_dir=None
) -> tuple[tuple[str, pd.DataFrame], tuple[str, pd.DataFrame]]:
    """Run instance analysis on the given local path to instances."""
    # Set the 'mix' column to the value of the 'model' column
    logger.info(f"Loaded {len(df):,} model predictions")

    logger.info(models)

    task_names = []  # e.g., mmlu:rc
    task_aliases = []  # e.g., [mmlu_abstract_algebra:rc::olmes, ...]
    metrics = []  # e.g., primary_score
    for task in tasks:
        # Expand the named group if it exists
        task_alias = ALL_NAMED_GROUPS.get(task, task)

        # Add setup for primary_score
        metrics += ["primary_score"]
        task_aliases += [task_alias]
        task_names += [task]

        # Add setup for bpb, if it exists
        bpb_task_name = make_bpb_name(task)
        if bpb_task_name is not None:
            metrics += ["logits_per_byte_corr"]
            task_names += [bpb_task_name]
            task_aliases += [task_alias]

    results = []
    with tqdm(total=len(tasks)) as pbar:
        for task, task_alias, metric in zip(task_names, task_aliases, metrics):
            pbar.set_description(f"Computing paired permutation test on {len(models)} models for {task}")

            N_COLS = 2
            N_ROWS = 1

            fig, axes = plt.subplots(
                N_ROWS,
                N_COLS // 2,
                gridspec_kw={"wspace": 0.4, "hspace": 0},
                figsize=(0.4 * len(models) * (N_COLS // 2), 0.85 * len(models) * (N_ROWS * 2)),
                squeeze=False,
            )

            result = run_paired_comparison(
                df, task=task, task_alias=task_alias, metric=metric, model_names=models, ax=axes[0, 0]
            )

            results.append(result)

            if any(ax.has_data() for row in axes for ax in row) and render_plots:
                logger.info(f"Saving figure(s) for {task}...")
                assert plot_dir is not None, plot_dir
                os.makedirs(plot_dir, exist_ok=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    fig.tight_layout()
                plt.savefig(
                    # bpt: bootstrap permutation test
                    plot_dir / f"bpt_{task.replace(':', '_')}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )

            plt.close()
            pbar.update(1)

    # Remove tasks with empty results
    results = [item for item in results if item]

    # Sort by task name
    results.sort(key=lambda x: x["task"])

    output_columns = [
        "scores",
        "p_values",
        "metric",
        "models",
        "task",
    ]

    return pd.DataFrame(results, columns=output_columns)
