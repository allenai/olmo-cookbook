import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
from cookbook.analysis.stats import compute_significance
from cookbook.analysis.utils.constants_tasks import get_title_from_task

def run_paired_comparison(df, task, model_names, metric='primary_score', axes=None):
    task_name = get_title_from_task(task)

    model_names = sorted(list(set(model_names)))

    ax: plt.Axes = axes if axes is not None else None
    _, p_values, _ = compute_significance(
        df, 
        models=model_names, 
        metric=metric,  
        step=None,  # the models have different checkpoint steps
        last_n=1,   # the "last n" checkpoints to average results
        alpha=0.05, # significance level
        tasks=[task],
        do_plot=[ax],
        plot_sig_clusters=False,
        quiet=True
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