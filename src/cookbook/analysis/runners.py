import logging
import os
import sys
import warnings
from typing import Optional

sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from cookbook.analysis.constants import (
    get_title_from_task,
)
from cookbook.analysis.stats import compute_significance

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REVERSED_METRICS = [
    "margin_per_byte",
    "norm_correct_prob_per_byte",
    "correct_prob_per_byte",
    "correct_logit_per_byte",
    "logits_per_char_corr",
    "logits_per_byte_corr",
    "bits_per_byte_corr",
]


def run_paired_comparison(
    df: pd.DataFrame,
    task: list[list[str] | str],
    model_names: list[str],
    axes: plt.Axes,
    metric: str = "primary_score",
) -> dict:
    task_name = get_title_from_task(task)
    model_names = sorted(list(set(model_names)))
    ax: Optional[plt.Axes] = axes

    _, p_values = compute_significance(
        df,
        models=model_names,
        metric=metric,
        alpha=0.05,  # significance level
        tasks=task,
        plot_axes=ax,
        plot_sig_clusters=False,
        quiet=True,
    )

    # Return nothing if test failed to run
    if task_name not in p_values:
        return {}

    mixes, scores, p_values, sig_clusters = p_values[task_name]

    return {
        "mixes": mixes,
        "scores": scores,
        "p_values": p_values,
        "sig_clusters": sig_clusters,
        "task_name": task_name,
        "task_set": task,
        "metric": metric,
        "model_names": model_names,
    }


def run_instance_analysis(
    df, render_plots=True, plot_dir=None
) -> tuple[tuple[str, pd.DataFrame], tuple[str, pd.DataFrame]]:
    """Run instance analysis on the given local path to instances."""
    # Set the 'mix' column to the value of the 'model' column
    df = df.reset_index()
    df = df.set_index(["alias", "model_name"])

    logger.info(f"Loaded {len(df):,} model predictions")

    ALL_MODELS = sorted(df.index.get_level_values("model_name").unique().to_list())
    ALL_TASKS = sorted(df.index.get_level_values("alias").unique().to_list())

    logger.info(ALL_TASKS)
    logger.info(ALL_MODELS)

    # tasks = get_task_sets(ALL_TASKS)
    tasks = ALL_TASKS
    # named_tasks = [get_title_from_task(task_set) for task_set in tasks]

    # Negate metrics where lower is better, so the ordering is the same
    for col in REVERSED_METRICS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: -x if pd.notna(x) else x)

    primary = []
    bpb = []
    with tqdm(total=len(tasks)) as pbar:
        for task in tasks:
            task_name = get_title_from_task(task)
            pbar.set_description(f"Computing paired permutation test on {len(ALL_MODELS)} models for {task_name}")

            N_COLS = 2
            N_ROWS = 1

            fig, axes = plt.subplots(
                N_ROWS * 2,
                N_COLS // 2,
                gridspec_kw={"wspace": 0.4, "hspace": 0},
                figsize=(0.4 * len(ALL_MODELS) * (N_COLS // 2), 0.85 * len(ALL_MODELS) * (N_ROWS * 2)),
                squeeze=False,
            )

            primary.append(
                run_paired_comparison(
                    df, task=[task], model_names=ALL_MODELS, metric="primary_score", axes=axes[0, 0]
                )
            )

            bpb.append(
                run_paired_comparison(
                    df, task=[task], model_names=ALL_MODELS, metric="bits_per_byte_corr", axes=axes[1, 0]
                )
            )

            if any(ax.has_data() for row in axes for ax in row) and render_plots:
                logger.info(f"Saving figure(s) for {task_name}...")
                assert plot_dir is not None, plot_dir
                os.makedirs(plot_dir, exist_ok=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    fig.tight_layout()
                plt.savefig(
                    plot_dir / f"paired_comparison_{task_name.replace(':', '_')}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )

            plt.close()
            pbar.update(1)

    # Remove tasks with empty results
    primary = [item for item in primary if item]
    bpb = [item for item in bpb if item]

    # Sort by task name
    primary.sort(key=lambda x: x["task_name"])
    bpb.sort(key=lambda x: x["task_name"])

    output_columns = [
        "scores",
        "p_values",
        "metric",
        "models",
        "task_name",
    ]

    return (
        (
            "primary_score",
            pd.DataFrame(primary, columns=output_columns),
        ),
        (
            "bits_per_byte_corr",
            pd.DataFrame(bpb, columns=output_columns),
        ),
    )
