import os, sys, warnings
sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
from tqdm import tqdm

from cookbook.analysis.stats import compute_significance
from cookbook.analysis.dataloader import get_nd_array
from cookbook.analysis.process.hf import pull_predictions_from_hf
from cookbook.analysis.utils.constants_tasks import get_title_from_task
from cookbook.analysis.utils import PLOT_DIR

REVERSED_METRICS = ['margin_per_byte', 'norm_correct_prob_per_byte', 'correct_prob_per_byte', 'correct_logit_per_byte', 'logits_per_char_corr', 'logits_per_byte_corr', 'bits_per_byte_corr']

# Optional color map for families of models
COLOR_MAP = {
    'Qwen': 'green',
    'Llama': 'blue',
    'LLaMA': 'blue',
    'Mistral': 'orange',
    '3B': 'black',
    'OLMo': 'pink',
    'pythia': 'brown',
    'gemma': 'teal',
    'phi': 'black',
    'deepseek': 'pink',
    'zephyr': 'green',
    'neo': 'orange',
    'falcon': 'blue',

    # code models 
    'starcoder': 'grey',
    'stablelm': 'grey',
}


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


def is_excluded_external_model(m):
    """ Models excluded from external base evals in allenai/ladder-evals """
    OLL2_INSTRUCT_MODELS = [
        # These are models on the OLL2 leaderboard that are actually instruct models
        'instruct',
        'superthoughts',
        'helpingai',
        'fox',
        'llmchat',
        'intern',
        'magistrate', # legal annealing
        'fietje', # phi fine-tune
        'llama-3-6.3b', # pruned llama 3
        'loxa', # very suspicious
        'llumix', # hungarian instruction tune
        'yarm', # instruction tune for context
        'lucie', # looks suspicious, i really think they snuck in instruct data here
        'nepali',
        'windy',
        'yarn' # long context fine-tuned models
        'llama-160m', # these models are just really bad
        'llama-43m',
        'llama-68m',

        # missing code evals
        'salamandra',
        'llama-160m',
    ]

    # These models have broken or incomplete results
    if 'Minitron' in m or \
        'Mistral' in m or \
        'bloom' in m or \
        'granite' in m or \
        'pruned' in m or \
        'INTELLECT-1' in m or \
        'TinyYi-7B-Test' in m or \
        'InstructLM-500M' in m or \
        'Qwen1.5-MoE' in m:
        return True
    if any(name in m.lower() for name in OLL2_INSTRUCT_MODELS):
        return True
    return False


def run_instance_analysis(local_path_instances):
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

            # Paried permutation test
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


def run_benchmark_analysis(local_path_metrics, external_evals_path):
    # Load external evals
    local_path_external_evals = pull_predictions_from_hf(external_evals_path, "benchmarks")
    df_external = pd.read_parquet(local_path_external_evals)
    print(f'Loaded {len(df_external):,} external model evaluations')

    # Get a list of models to exclude from our "external models"
    EXTERNAL_MODELS = list(df_external['model'].unique())
    EXTERNAL_TASKS = sorted(list(df_external['task'].unique()))
    all_external_models = sorted([
        model for model in EXTERNAL_MODELS 
        if model not in
            [model for model in EXTERNAL_MODELS if 'xC' in model] + # exclude 1B-5xC models
            [model for model in EXTERNAL_MODELS if 'peteish-moreeval' in model] + # exclude ladder models
            ['peteish13-highlr'] # exclude intermediate checkpoints from 13B
        and not is_excluded_external_model(model)
    ])

    # Load internal evals
    df_internal = pd.read_parquet(local_path_metrics)
    print(f'Loaded {len(df_external):,} internal model evaluations')
    
    all_internal_models = list(df_internal['model'].unique())
    INTERNAL_TASKS = sorted(list(df_internal['task'].unique()))

    selected_tasks = get_task_sets(INTERNAL_TASKS)

    # Render figures
    results = []
    with tqdm(total=len(selected_tasks)) as pbar:
        for task in selected_tasks:
            pbar.set_description(f"Plotting results on {len(all_external_models)} external models for {get_title_from_task(task)}")

            # Create scatter plot of metrics
            N_COLS = 1
            N_ROWS = 1
            fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(12*N_COLS, 8*N_ROWS), squeeze=False) # 0.35

            ax: plt.Axes = axes[0, 0]
            
            metric_names = ["logits_per_byte_corr", "primary_score"]
            x_metric, y_metric = metric_names[0], metric_names[1]
            task_name = get_title_from_task(task)
            
            all_x = []
            all_y = []
            model_labels = []
            colors = []
            
            # Process external models first
            for model in all_external_models:
                _, scores = get_nd_array(df_external, "model", metric_names, model=model, task=task, step=None)
                # average over tasks, if applicable
                if scores.ndim > 1:
                    scores = scores.mean(axis=-1)
                if scores is not None and len(scores) > 0:
                    color = 'red'  # Default color for external
                    for key, value in COLOR_MAP.items():
                        if key in model:
                            color = value
                            break
                    
                    all_x.append(scores[0])
                    all_y.append(scores[1])
                    model_labels.append(model)
                    colors.append(color)
            
            # Plot external models with X marker
            ax.scatter(all_x, all_y, marker='x', s=3, c=colors)
            
            # Now process and plot internal models with stars
            internal_x = []
            internal_y = []
            internal_labels = []
            for model in all_internal_models:
                _, scores = get_nd_array(df_internal, "model", metric_names, model=model, task=task, step=None)
                # average over tasks, if applicable
                if scores.ndim > 1:
                    scores = scores.mean(axis=-1)
                if scores is not None and len(scores) > 0:
                    internal_x.append(scores[0])
                    internal_y.append(scores[1])
                    internal_labels.append(model)
                    
            # Plot internal models with large golden stars
            ax.scatter(internal_x, internal_y, marker='*', s=50, c='gold', edgecolor='black')
            
            # Add internal models to the overall lists for labeling
            all_x.extend(internal_x)
            all_y.extend(internal_y)
            model_labels.extend(internal_labels)
            colors.extend(['gold'] * len(internal_labels))
            
            ax.set_xlabel(x_metric)
            ax.set_ylabel(y_metric) 
            ax.set_title(f"{task_name} {x_metric} -> {y_metric}")
            
            # Add model name labels to points
            texts = []
            for i, model in enumerate(model_labels):
                texts += [ax.text(all_x[i], all_y[i], model, fontsize=5, alpha=0.8, ha="center", va="center")]

            # Adjust text annotations to not overlap with each other
            import matplotlib
            # Remove existing annotation
            existing_annotations = [
                child for child in ax.get_children() if isinstance(child, matplotlib.text.Annotation)
            ]
            for child in existing_annotations:
                child.remove()
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
                avoid_points=True,
                avoid_self=True,
                avoid_lines=True,
                existing_annotations=existing_annotations,
                autoalign="xy",
                force_points=0.7,
                force_text=0.3,
                expand_points=(1.7, 2.0),
                expand_text=(1.1, 2.0),
                ax=ax,
            )

            if any(ax.has_data() for row in axes for ax in row):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    fig.tight_layout()
                os.makedirs(PLOT_DIR, exist_ok=True)
                plt.savefig(f'{PLOT_DIR}/performance_{get_title_from_task(task)}.pdf', format='pdf', bbox_inches='tight')
            plt.close()

            pbar.update(1)
