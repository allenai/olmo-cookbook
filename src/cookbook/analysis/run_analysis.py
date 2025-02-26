import os, sys, warnings
sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from cookbook.analysis.stats import compute_significance
from cookbook.analysis.utils.constants_tasks import get_title_from_task
from cookbook.analysis.utils import PLOT_DIR


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

REVERSED_METRICS = ['margin_per_byte', 'norm_correct_prob_per_byte', 'correct_prob_per_byte', 'correct_logit_per_byte', 'logits_per_char_corr', 'logits_per_byte_corr', 'bits_per_byte_corr']


def task_tag(task_name):
    return task_name.split("::")[0].replace(":rc", "") if "::" in task_name else task_name


def get_task_sets(all_tasks):
    mmlu      = [t for t in all_tasks if 'mmlu' in t and ':' not in t and '_pro_' not in t]
    minerva   = [t for t in all_tasks if 'minerva' in t and ':' not in t]
    mmlu_pro  = [t for t in all_tasks if '_pro_' in t and ':rc' in t]
    mmlu_mc   = [t for t in all_tasks if 'mmlu' in t and ':mc' in t and '_pro_' not in t]
    olmes     = ['arc_challenge', 'arc_easy', 'boolq', 'csqa', 'hellaswag', 'openbookqa', 'piqa', 'socialiqa', 'winogrande']
    olmes_mc  = [f'{task}:mc' for task in olmes]
    olmes_gen = ['drop', 'gsm8k', 'jeopardy', 'naturalqs', 'squad', 'triviaqa']
    agi_eval  = [t for t in all_tasks if 'agi_eval' in t and ':' not in t]
    bbh       = [t for t in all_tasks if 'bbh' in t and ':' not in t]
    paloma    = [t for t in all_tasks if 'paloma' in t]

    selected_tasks = olmes + olmes_mc + olmes_gen + [olmes, mmlu, olmes_mc, mmlu_mc, olmes_gen, minerva, agi_eval, bbh, paloma]
    selected_tasks += ['gsm_symbolic_main', mmlu_pro, 'autobencher', 'autobencher:mc']
    selected_tasks += ['minerva_math_500', 'mbpp', 'mbppplus', 'codex_humaneval', 'codex_humanevalplus', 'copycolors:mc']

    selected_tasks = [task for task in selected_tasks if len(task) > 0] # exclude tasks where we don't see them in the df

    return selected_tasks


def run_analysis(local_path_instances):
    print(f'Loading {local_path_instances}')

    df = pd.read_parquet(local_path_instances)

    # Set the 'mix' column to the value of the 'model' column
    df = df.reset_index()
    df['mix'] = df['model']
    df['step'] = df['step'].fillna('0')
    df = df.set_index(['task', 'model', 'step', 'mix'])

    print(f'Loaded {len(df):,} model evaluations')

    MODELS = sorted(df.index.get_level_values('model').unique().to_list())
    TASKS  = sorted(df.index.get_level_values('task').unique().to_list())

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
            pbar.set_description(f"Computing paired permutation test on {len(MODELS)} models for {get_title_from_task(task)}")

            N_COLS = 2
            N_ROWS = 1
            fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(0.85*len(MODELS)*N_COLS, 0.25*len(MODELS)*N_ROWS), squeeze=False) # 0.35
            
            result = run_paired_comparison(
                df, 
                task=task, 
                model_names=MODELS,
                metric='primary_score', 
                axes=axes[0, 0]
            )
            results += [result]

            result = run_paired_comparison(
                df, 
                task=task, 
                model_names=MODELS,
                metric='bits_per_byte_corr',
                axes=axes[0, 1]
            )
            results += [result]

            if any(ax.has_data() for row in axes for ax in row):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    fig.tight_layout()
                os.makedirs(PLOT_DIR, exist_ok=True)
                plt.savefig(f'{PLOT_DIR}/paired_comparison_{get_title_from_task(task)}.pdf', format='pdf', bbox_inches='tight')
            plt.close()

            pbar.update(1)

    results_df = pd.DataFrame(results, index=task_names)

    return results_df