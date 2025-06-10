import json
import numpy as np

import click
from rich.console import Console
from rich.table import Table

from cookbook.cli.eval import get_results
from cookbook.eval.aggregation import compute_pca, compute_kendall_tau, grid_search_optimizer, neg_kendall_tau, remove_incomplete_tasks, results_to_ndarray

@click.option(
    "-d", "--dashboard", type=str, required=True, help="Set dashboard name"
)
@click.option(
    "-m",
    "--models",
    type=str,
    multiple=True,
    default=None,
    help="Set specific models to show. Can be specified multiple times.",
)
@click.option(
    "-tl",
    "--tasks-dev-target",
    type=str,
    required=True,
    multiple=True,
    help="Tasks to compute PCA over.",
)
@click.option(
    "-ts",
    "--tasks-dev-small",
    type=str,
    required=False,
    multiple=True,
    help="Small-scale tasks to maximize rank correlation for the PC-1 of --tasks-dev.",
)
@click.option(
    "-P",
    "--partial",
    is_flag=True,
    help="Compute weights even if not all data is available.",
)
def compute_macro_avg_weights(
    dashboard: str,
    models: list[str],
    tasks_dev_target: list[str],
    tasks_dev_small: list[str],
    partial: bool,
) -> None:
    if not tasks_dev_small:
        tasks_dev_small = tasks_dev_target

    # Get results using dashboard
    results_small = get_results(
        dashboard=dashboard,
        models=models,
        tasks=tasks_dev_small,
        
        # defaults
        format="return", sort_by="column", sort_column_name="", sort_descending=False,
        force=False, skip_on_fail=True, missing_pairs=False,
    )

    results_large = get_results(
        dashboard=dashboard,
        models=models,
        tasks=tasks_dev_target,
        
        # defaults
        format="return", sort_by="column", sort_column_name="", sort_descending=False,
        force=False, skip_on_fail=True, missing_pairs=False,
    )

    # Get intersection of all model sets among both results
    task_model_sets = []
    for task_results in [results_small, results_large]:
        for task in task_results:
            task_model_sets.append(set(task_results[task].keys()))
    all_models = sorted(set.intersection(*task_model_sets))

    small_tasks, results_small_np = results_to_ndarray(results_small, all_models) # (task, model)
    large_tasks, results_large_np = results_to_ndarray(results_large, all_models) # (task, model)

    # Remove any rows in results_small_np that contain None values
    small_tasks, results_small_np = remove_incomplete_tasks(small_tasks, results_small_np, partial=partial)
    large_tasks, results_large_np = remove_incomplete_tasks(large_tasks, results_large_np, partial=partial)
    
    # Compute simple macro-average
    results_large_macro_avg = np.mean(results_large_np, axis=0) # (model,)

    # Compute PC-1 for the large-scale tasks
    results_large_np_T = results_large_np.T # (model, task)
    data_centered, weights, explained_variance_ratio = compute_pca(results_large_np_T)
    pca_results = np.dot(data_centered, weights)
    results_large_pc_1 = pca_results[:, 0] # get PC-1

    # Rescale so each PC's weights sum to 1
    rescaled_weights = weights / weights.sum(axis=0)

    print(f'Fit PC-1 to explain {explained_variance_ratio[0]*100:.2f}% of variance')
    
    # Print weights of PCs
    table = Table()
    table.add_column("Target Task")
    for i in range(min(5, rescaled_weights.shape[1])):
        table.add_column(f"PC-{i+1}")
    sorted_indices = np.argsort(rescaled_weights[:,0])[::-1]  # Sort by PC-1 descending
    for idx in sorted_indices:
        row = [large_tasks[idx]]
        for i in range(min(5, rescaled_weights.shape[1])):
            row.append(f"{rescaled_weights[idx,i]:.3f}")
        table.add_row(*row)
    console = Console()
    console.print(table)

    # Compute optimal weights with grid search
    w_init = np.ones(results_small_np.shape[0]) / results_small_np.shape[0]
    w_optimal, _ = grid_search_optimizer(neg_kendall_tau, w_init, results_small_np, results_large_pc_1)

    # Compute new weighted avg
    small_uniform_macro_avg = np.mean(results_small_np, axis=0) # (model,)
    results_small_weighted_avg = np.sum(w_optimal[:, None] * results_small_np, axis=0) # (model,)

    print("Computed weights of small-scale tasks to predict PC-1:")

    # Print optimal weights to maximize corr(small scale, PC-1)
    table = Table()
    table.add_column("Small-scale Task")
    table.add_column("Weight")
    sorted_pairs = sorted(zip(small_tasks, w_optimal), key=lambda x: x[1], reverse=True)
    for task, weight in sorted_pairs:
        table.add_row(task, f"{weight:.3f}")
    console = Console()
    console.print(table)

    # Print PC-1 and macro average scores across models
    table = Table()
    table.add_column("Model")
    table.add_column("Target Macro Avg")
    table.add_column("Target PC-1") 
    table.add_column("Small Macro Avg")
    table.add_column("Small Weighted Avg")
    sorted_indices = np.argsort(results_small_weighted_avg)[::-1]  # Sort descending
    for idx in sorted_indices:
        table.add_row(
            all_models[idx],
            f"{results_large_macro_avg[idx]:.3f}",
            f"{results_large_pc_1[idx]:.3f}",
            f"{small_uniform_macro_avg[idx]:.3f}",
            f"{results_small_weighted_avg[idx]:.3f}"
        )
    console = Console()
    console.print(table)

    # Compute kendal tau of (baseline) simple macro avg
    decision_acc, kendall_tau = compute_kendall_tau(small_uniform_macro_avg, results_large_pc_1)
    print("Uniform small-scale macro avg: ", f"\n\tDecision accuracy: {decision_acc:.2f}", f"\n\tKendall Tau: {kendall_tau:.2f}")
    
    decision_acc, kendall_tau = compute_kendall_tau(results_small_weighted_avg, results_large_pc_1)
    print("Weighted small-scale avg: ", f"\n\tDecision accuracy: {decision_acc:.2f}", f"\n\tKendall Tau: {kendall_tau:.2f}")
    
    print()

    # Output the "small" fitted task weights
    print(json.dumps(dict(sorted(zip(small_tasks, w_optimal.tolist())))))
