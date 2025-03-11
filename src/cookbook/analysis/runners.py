import os
import sys
import warnings

sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from cookbook.analysis.stats import compute_significance
from cookbook.analysis.constants import (
    PLOT_DIR,
    get_task_sets,
    get_title_from_task,
)


REVERSED_METRICS = [
    "margin_per_byte",
    "norm_correct_prob_per_byte",
    "correct_prob_per_byte",
    "correct_logit_per_byte",
    "logits_per_char_corr",
    "logits_per_byte_corr",
    "bits_per_byte_corr",
]


def run_paired_comparison(df, task, model_names, metric="primary_score", axes=None):
    task_name = get_title_from_task(task)

    model_names = sorted(list(set(model_names)))

    from typing import Optional

    ax: Optional[plt.Axes] = axes if axes is not None else None
    sig_values, p_values = compute_significance(
        df,
        models=model_names,
        metric=metric,
        step=None,  # the models have different checkpoint steps
        last_n=1,  # the "last n" checkpoints to average results
        alpha=0.05,  # significance level
        tasks=[task],
        plot_axes=ax,
        plot_sig_clusters=False,
        quiet=True,
    )

    if ax:
        # (plotting logic goes here)
        # ax.set_ylabel(primary_score_name)
        pass

    # Return nothing if test failed to run
    if task_name not in p_values:
        return {}

    mixes, scores, p_values, sig_clusters = p_values[task_name]

    return {
        # Add return values here, if needed!
        # "test": 0
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


def run_instance_analysis(local_path_instances):
    print(f"Loading {local_path_instances}")

    df = pd.read_parquet(local_path_instances)

    # Set the 'mix' column to the value of the 'model' column
    df = df.reset_index()
    df["mix"] = df["model"]
    df["step"] = df["step"].fillna("0")
    df = df.set_index(["task", "model", "step", "mix"])

    print(f"Loaded {len(df):,} model evaluations")

    MODELS = sorted(df.index.get_level_values("model").unique().to_list())
    TASKS = sorted(df.index.get_level_values("task").unique().to_list())

    selected_tasks = get_task_sets(TASKS)
    # selected_tasks = TASKS

    task_names = [get_title_from_task(task) for task in selected_tasks]

    # Negate metrics where lower is better, so the ordering is the same
    for col in REVERSED_METRICS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: -x if pd.notna(x) else x)

    # Render figures
    results = []
    with tqdm(total=len(selected_tasks)) as pbar:
        for task in selected_tasks:
            pbar.set_description(
                f"Computing paired permutation test on {len(MODELS)} models for {get_title_from_task(task)}"
            )

            # Paried permutation test
            N_COLS = 2
            N_ROWS = 1
            fig, axes = plt.subplots(
                N_ROWS, N_COLS, figsize=(0.85 * len(MODELS) * N_COLS, 0.25 * len(MODELS) * N_ROWS), squeeze=False
            )  # 0.35

            result = run_paired_comparison(
                df, task=task, model_names=MODELS, metric="primary_score", axes=axes[0, 0]
            )
            results += [result]

            result = run_paired_comparison(
                df, task=task, model_names=MODELS, metric="bits_per_byte_corr", axes=axes[0, 1]
            )
            results += [result]

            if any(ax.has_data() for row in axes for ax in row):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    fig.tight_layout()
                os.makedirs(PLOT_DIR, exist_ok=True)
                plt.savefig(
                    f"{PLOT_DIR}/paired_comparison_{get_title_from_task(task)}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
            plt.close()

            pbar.update(1)

    results_df = pd.DataFrame(results, index=task_names)

    return results_df
