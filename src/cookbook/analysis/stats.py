from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from cookbook.analysis.plot.heatmap import plot_heatmap
from cookbook.analysis.constants import get_title_from_task
from cookbook.analysis.utils.dataloader import get_nd_array, get_slice
from cookbook.analysis.utils.pce import (
    compute_pairwise_p_values,
    compute_weighted_pairwise_p_values,
)


def perc_significant(p_values, alpha=0.05):
    """Calculate the % of statistically significant comparisons"""
    return ((p_values > (1 - alpha)).sum() + (p_values < alpha).sum()) / (~np.isnan(p_values)).sum()


def calculate_standard_error(avg_score, num_scores):
    """https://arxiv.org/pdf/2411.00640#page=2.55"""
    return np.sqrt((avg_score * (1 - avg_score)) / num_scores)


def get_bound(arr, idx, alpha):
    """Get the first index where arr[i] < alpha"""
    condition = arr < alpha  # | (arr > (1-alpha))
    indices = np.argwhere(condition)

    if indices.size > 0:
        first_index = tuple(indices[0])
    else:
        return len(arr) + idx

    return first_index[0] + idx


def create_stratified_array(counts):
    """Convert counts to 1D array of weights: [1172, 2304] => [1172, 1172, ..., 2304, 2304, ...]"""
    return np.concatenate([np.full(count, count, dtype=np.float64) for value, count in enumerate(counts)])


def get_sig_cluster_bound(p_vals, idx, alpha, conservative: Optional[bool] = False):
    """Given an idx and p_vals, compute the boundary of the next significance cluster"""
    col = p_vals[:, idx][idx + 1 :]
    row = p_vals[idx, :][idx + 1 :]
    bounds = get_bound(col, idx, alpha=alpha), get_bound(row, idx, alpha=alpha)
    if conservative:
        return max(bounds) + 1  # conservative clustering (gives less clusters)
    else:
        return min(bounds) + 1  # PERM-INPUTS clustering


def get_sig_clusters(p_vals, alpha=0.01):
    """
    Start with highest scoring mix, assign rank 1 to all mixes until we have
    encountered a mix statistically significantly different from any mix so far.
    """
    sig_clusters = np.zeros(p_vals.shape[0])

    curr, curr_cluster = 0, 0
    while curr < p_vals.shape[0]:
        idx = curr
        cluster_bound = get_sig_cluster_bound(p_vals, idx, alpha)
        for _ in range(idx, cluster_bound):
            sig_clusters[curr] = curr_cluster
            curr += 1
        curr_cluster += 1

    return sig_clusters


def compute_significance(
    df: DataFrame,
    models,
    metric: str,
    step: Optional[str] = "max",
    last_n: int = 1,
    tasks: Optional[list[str]] = None,
    alpha: float = 0.05,
    num_permutations: int = 1_000,
    plot_axes: Optional[np.ndarray] = None,
    pretty_mix_names: Optional[dict[str, str]] = None,
    plot_sig_clusters: bool = True,
    plot_clean: bool = False,
    quiet: bool = False,
) -> tuple[DataFrame, dict]:
    if tasks is None:
        tasks = df.index.get_level_values("task").unique().tolist()

    sig_results = pd.DataFrame(index=["perc_sig"], columns=tasks)
    all_p_values = {}

    n_tasks = len(tasks)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(0.5 * len(models), 0.4 * len(models) * n_tasks))
    if n_tasks == 1:
        axes = [axes]

    for i, task in tqdm(enumerate(tasks), desc="Computing pairwise comparisons", total=len(tasks), disable=quiet):
        if last_n > 1:
            assert step == "max"

            mixes, scores = get_nd_array(df, ["mix", "step"], metric, model=models, task=task)
            scores = scores[:, -last_n:, :]  # get last n steps
            scores = scores.mean(axis=1)  # average over last n ckpts

            # Recover just the mix names
            mixes = np.array([name for name, step in mixes])

            # Sort based on new aggregate (average/concat)
            mix_sums = scores.sum(axis=1)
            sorted_indices = mix_sums.argsort()[::-1]
            mixes = mixes[sorted_indices].tolist()
            scores = scores[sorted_indices]
        else:
            mixes, scores = get_nd_array(df, "mix", metric, model=models, task=task, step=step, sorted=True)

        if len(scores) == 0:
            print(f"Found no scores for {get_title_from_task(task)}! Skipping...")
            continue

        if isinstance(task, list):
            # Get value counts for each task
            slices = get_slice(df, model=models, task=task)
            unique_counts = slices.groupby("task")["native_id"].nunique()
            weights = create_stratified_array(unique_counts)

            # Compute paired permutation test with instance weights
            p_values, mix_scores, _ = compute_weighted_pairwise_p_values(
                scores, num_permutations=num_permutations, weights=weights, return_scores=True
            )

            # Reorder both rows and columns of p_values
            sorted_indices = np.argsort(mix_scores)[::-1]
            p_values = p_values[np.ix_(sorted_indices, sorted_indices)]
            mixes = np.array(mixes)[sorted_indices]
            mix_scores = mix_scores[sorted_indices]
            p_values[np.tril_indices_from(p_values, k=-1)] = np.nan

            # Change task name
            task = get_title_from_task(task)
        else:
            p_values, mix_scores, _ = compute_pairwise_p_values(
                scores, num_permutations=num_permutations, return_scores=True
            )

        sig_clusters = None
        if plot_sig_clusters:
            sig_clusters = get_sig_clusters(p_values, alpha=alpha)

        perc_sig = perc_significant(p_values, alpha=alpha)
        sig_results.loc["perc_sig", task] = perc_sig
        all_p_values[task] = (mixes, scores, p_values, sig_clusters)

        if plot_axes:
            if pretty_mix_names is not None:
                mix_names = [pretty_mix_names[mix] for mix in mixes]
            else:
                mix_names = mixes
            axes[i] = plot_heatmap(
                axes[i], p_values, mix_names, mix_scores, sig_clusters, alpha=alpha, plot_clean=plot_clean
            )
            title = (
                r"$p$"
                + f'-values for {task} (n={scores.shape[1]}) {("last " + str(last_n) + " steps" if last_n > 1 else "")}({metric}), perc sig={(perc_sig*100):.2f}%'
            )
            if len(models) < 15:
                title = r"$p$" + f"-values for {task}, perc sig={(perc_sig*100):.2f}%"
            axes[i].set_title(title, fontsize=10)

        fig.tight_layout()

    return sig_results, all_p_values
