import os
import sys
import warnings
from typing import Optional

sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from cookbook.analysis.constants import (
    PLOT_DIR,
    get_task_sets,
    get_title_from_task_set,
)
from cookbook.analysis.stats import compute_significance

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
    task_set: list[str] | list[list[str]],
    model_names: list[str],
    axes: plt.Axes,
    metric: str = "primary_score",
) -> dict:
    task_name = get_title_from_task_set(task_set)
    model_names = sorted(list(set(model_names)))
    ax: Optional[plt.Axes] = axes

    _, p_values = compute_significance(
        df,
        models=model_names,
        metric=metric,
        step=None,  # the models have different checkpoint steps
        last_n=1,  # the "last n" checkpoints to average results
        alpha=0.05,  # significance level
        tasks=task_set,
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
        "task_set": task_set,
        "metric": metric,
        "model_names": model_names,
    }


def task_tag(task_name):
    return task_name.split("::")[0].replace(":rc", "") if "::" in task_name else task_name


def is_excluded_external_model(m):
    """Models excluded from external base evals in allenai/ladder-evals"""
    OLL2_INSTRUCT_MODELS = [
        # These are models on the OLL2 leaderboard that are actually instruct models
        "instruct",
        "superthoughts",
        "helpingai",
        "fox",
        "llmchat",
        "intern",
        "magistrate",  # legal annealing
        "fietje",  # phi fine-tune
        "llama-3-6.3b",  # pruned llama 3
        "loxa",  # very suspicious
        "llumix",  # hungarian instruction tune
        "yarm",  # instruction tune for context
        "lucie",  # looks suspicious, i really think they snuck in instruct data here
        "nepali",
        "windy",
        "yarn"  # long context fine-tuned models
        "llama-160m",  # these models are just really bad
        "llama-43m",
        "llama-68m",
        # missing code evals
        "salamandra",
        "llama-160m",
    ]

    # These models have broken or incomplete results
    if (
        "Minitron" in m
        or "Mistral" in m
        or "bloom" in m
        or "granite" in m
        or "pruned" in m
        or "INTELLECT-1" in m
        or "TinyYi-7B-Test" in m
        or "InstructLM-500M" in m
        or "Qwen1.5-MoE" in m
    ):
        return True
    if any(name in m.lower() for name in OLL2_INSTRUCT_MODELS):
        return True
    return False


def run_instance_analysis(local_path_instances) -> tuple[tuple[str, pd.DataFrame], tuple[str, pd.DataFrame]]:
    """Run instance analysis on the given local path to instances."""
    print(f"Loading {local_path_instances}")

    df = pd.read_parquet(local_path_instances)

    # Set the 'mix' column to the value of the 'model' column
    df = df.reset_index()
    df["mix"] = df["model"]
    df["step"] = df["step"].fillna("0")
    df = df.set_index(["task", "model", "step", "mix"])

    print(f"Loaded {len(df):,} model evaluations")

    ALL_MODELS = sorted(df.index.get_level_values("model").unique().to_list())
    ALL_TASKS = sorted(df.index.get_level_values("task").unique().to_list())

    task_sets = get_task_sets(ALL_TASKS)
    named_tasks = [get_title_from_task_set(task_set) for task_set in task_sets]

    # Negate metrics where lower is better, so the ordering is the same
    for col in REVERSED_METRICS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: -x if pd.notna(x) else x)

    primary = []
    bpb = []
    with tqdm(total=len(named_tasks)) as pbar:
        for task in task_sets:
            task_name = get_title_from_task_set(task)
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
                    df, task_set=[task], model_names=ALL_MODELS, metric="primary_score", axes=axes[0, 0]
                )
            )

            bpb.append(
                run_paired_comparison(
                    df, task_set=[task], model_names=ALL_MODELS, metric="bits_per_byte_corr", axes=axes[1, 0]
                )
            )

            if any(ax.has_data() for row in axes for ax in row):
                print(f"Saving figure(s) for {task_name}...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    fig.tight_layout()
                os.makedirs(PLOT_DIR, exist_ok=True)
                plt.savefig(
                    f"{PLOT_DIR}/paired_comparison_{task_name}.pdf",
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
