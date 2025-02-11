import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from cookbook.eval.process.hf import pull_predictions_from_hf
from cookbook.eval.metaanalysis import run_analysis
from cookbook.eval.ladder_wrapper import sort_experiment_names
from cookbook.eval.utils import ROOT_DIR, PLOT_DIR
from cookbook.eval.utils.constants_models import MODEL_NAMES_MIXES

from metaanalysis import get_title_from_task
plt.close()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore fitting warnings

BROKEN_MODELS = ["gemma-2b", "gemma-7b", "gemma-2-2b", "gemma-2-9b"] # gemma models broken in oe-eval


def task_tag(task_name):
    return task_name.split("::")[0].replace(":rc", "") if "::" in task_name else task_name


def get_selected_tasks(all_tasks):
    selected_tasks = [
        t
        for t in all_tasks
        if ":" not in t 
        and "bbh" not in t and "paloma" not in t and "llm_compression" not in t 
        and "custom_loss" not in t and "coqa" not in t and "deepmind" not in t
    ]

    selected_tasks += [
        t
        for t in all_tasks
        if ":mc" in t
        and "bbh" not in t and "paloma" not in t and "llm_compression" not in t
        and "custom_loss" not in t and "coqa" not in t and "deepmind" not in t
    ]

    mmlu      = [t for t in all_tasks if 'mmlu' in t and ':' not in t and '_pro_' not in t]
    minerva   = [t for t in all_tasks if 'minerva' in t and ':' not in t]
    mmlu_pro  = [t for t in all_tasks if '_pro_' in t and ':rc' in t]
    mmlu_mc   = [t for t in all_tasks if 'mmlu' in t and ':mc' in t and '_pro_' not in t]
    olmes     = ['arc_challenge', 'arc_easy', 'boolq', 'csqa', 'hellaswag', 'openbookqa', 'piqa', 'socialiqa', 'winogrande']
    olmes_mc  = [f'{task}:mc' for task in olmes]
    olmes_para        = [f'{task}:para' for task in olmes]
    olmes_distractors = [f'{task}:distractors' for task in olmes]
    olmes_enlarge     = [f'{task}:enlarge' for task in olmes]
    olmes_gen = ['drop', 'gsm8k', 'jeopardy', 'naturalqs', 'squad', 'triviaqa']
    agi_eval  = [t for t in all_tasks if 'agi_eval' in t and ':' not in t]
    bbh       = [t for t in all_tasks if 'bbh' in t and ':' not in t]
    paloma    = [t for t in all_tasks if 'paloma' in t]
    llm_compression = [t for t in all_tasks if 'llm_compression' in t]
    custom_loss = [t for t in all_tasks if 'custom_loss' in t]

    selected_tasks = olmes + olmes_gen + [olmes, olmes_para, olmes_distractors, olmes_enlarge, mmlu, olmes_mc, mmlu_mc, olmes_gen, minerva]
    selected_tasks += ['gsm_symbolic_main', mmlu_pro, 'autobencher', 'autobencher:mc']
    selected_tasks += ['minerva_math_500', 'mbpp', 'mbppplus', 'codex_humaneval', 'codex_humanevalplus', 'copycolors:mc']
    return selected_tasks


def main():
    local_path = pull_predictions_from_hf("allenai/ladder-evals", "benchmarks")

    df = pd.read_parquet(local_path)
    print(f'Loaded {len(df):,} model evaluations')

    # fix for the names of one of Ian's data mixes
    df.loc[df['mix'] == 'baseline', 'mix'] = 'dolma17'    

    MODELS = list(df['model'].unique())
    TASKS = sorted(list(df['task'].unique()))

    selected_tasks = get_selected_tasks(TASKS)

    # Ladder train models
    all_ladder_models = [model for model in MODELS if 'peteish-moreeval' in model]
    all_ladder_models = sort_experiment_names(all_ladder_models)

    # Eval models
    LLAMA_3_MODELS = [model for model in MODELS if 'Llama-3' in model]

    all_models = sorted([model for model in MODELS if model not in MODEL_NAMES_MIXES + BROKEN_MODELS + all_ladder_models])

    task_names = [get_title_from_task(task) for task in selected_tasks]
    
    # Render figures
    results = []
    for task in tqdm(selected_tasks):
        N_COLS = 4
        N_ROWS = 3
        fig, axes = plt.subplots(N_COLS, N_ROWS, figsize=(4*N_ROWS, 3*N_COLS), squeeze=False)
        result = run_analysis(
            df, 
            task=task, 
            ladder_models=all_ladder_models, 
            external_ladder_models=all_ladder_models + LLAMA_3_MODELS, 
            eval_ladder_models=[m for m in all_models if 'OLMo-7B-0424-hf' not in m],
            axes=axes
        )
        results += [result]
        fig.tight_layout()

        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(f'{PLOT_DIR}/metaanalysis_{get_title_from_task(task)}.pdf', format='pdf', bbox_inches='tight')

    results_df = pd.DataFrame(results, index=task_names)

    print(results_df)

if __name__ == '__main__': main()