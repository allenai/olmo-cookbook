import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from cookbook.analysis.process.hf import pull_predictions_from_hf
from cookbook.analysis.run_analysis import run_paired_comparison, get_title_from_task
from cookbook.analysis.scripts.pull_results import mirror_s3_to_local, process_local_folder
from cookbook.analysis.utils import DATA_DIR, PLOT_DIR
from cookbook.analysis.utils.constants_tasks import PRIMARY_METRICS_OLMES


olmes_gen = ['drop', 'gsm8k', 'jeopardy', 'naturalqs', 'squad', 'triviaqa']


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

    selected_tasks = olmes + olmes_gen + [olmes, mmlu, olmes_mc, mmlu_mc, olmes_gen, minerva]
    selected_tasks += ['gsm_symbolic_main', mmlu_pro, 'autobencher', 'autobencher:mc']
    selected_tasks += ['minerva_math_500', 'mbpp', 'mbppplus', 'codex_humaneval', 'codex_humanevalplus', 'copycolors:mc']

    selected_tasks = [task for task in selected_tasks if len(task) > 0] # exclude tasks where we don't see them in the df

    return selected_tasks


def run_analysis(local_path_instances):
    df = pd.read_parquet(local_path_instances)

    # Set the 'mix' column to the value of the 'model' column
    df = df.reset_index()
    df['mix'] = df['model']
    df['step'] = df['step'].fillna(0)
    df = df.set_index(['task', 'model', 'step', 'mix'])

    print(f'Loaded {len(df):,} model evaluations')

    MODELS = sorted(df.index.get_level_values('model').unique().to_list())
    TASKS  = sorted(df.index.get_level_values('task').unique().to_list())

    TASKS = sorted(TASKS)

    selected_tasks = get_selected_tasks(TASKS)
    # selected_tasks = TASKS

    task_names = [get_title_from_task(task) for task in selected_tasks]
    
    # Render figures
    results = []
    with tqdm(total=len(selected_tasks)) as pbar:
        for task in selected_tasks:
            pbar.set_description(f"Computing paired permutation test on {len(MODELS)} models for {get_title_from_task(task)}")

            primary_score_name = PRIMARY_METRICS_OLMES[task] if isinstance(task, str) and task in PRIMARY_METRICS_OLMES else 'primary_score'

            N_COLS = 2
            N_ROWS = 1
            fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(0.85*len(MODELS)*N_COLS, 0.25*len(MODELS)*N_ROWS), squeeze=False) # 0.35
            
            result = run_paired_comparison(
                df, 
                task=task, 
                model_names=MODELS,
                metric=('acc_per_char' if task not in olmes_gen else 'exact_match'), 
                axes=axes[0, 0]
            )
            results += [result]

            result = run_paired_comparison(
                df, 
                task=task, 
                model_names=MODELS,
                metric='logits_per_char_corr',
                axes=axes[0, 1]
            )
            results += [result]

            if any(ax.has_data() for row in axes for ax in row):
                fig.tight_layout()
                os.makedirs(PLOT_DIR, exist_ok=True)
                plt.savefig(f'{PLOT_DIR}/paired_comparison_{get_title_from_task(task)}.pdf', format='pdf', bbox_inches='tight')
            plt.close()

            pbar.update(1)

    results_df = pd.DataFrame(results, index=task_names)

    return results_df


def main():
    # local_path = pull_predictions_from_hf("allenai/ladder-evals", "benchmarks")
    # df.loc[df['mix'] == 'baseline', 'mix'] = 'dolma17' # # fix for the names of one of Ian's data mixes
    # MODELS = list(df['model'].unique())
    # TASKS = sorted(list(df['task'].unique()))

    # Pull pre-processed predictions from HF
    # predictions_local_path = pull_predictions_from_hf("allenai/peteish32-evals", "instances")
    # predictions_local_path = pull_predictions_from_hf("allenai/olmo2-anneals-evals", "main")

    # # Pull predictions from AWS
    # bucket_name = "ai2-llm"
    # s3_prefix = ["evaluation/anneal-peteish-7b"]
    # local_results_path = 'olmo2_anneals'

    # bucket_name = 'ai2-llm'
    # s3_prefix = 'evaluation/peteish32/'
    # local_results_path = 'peteish32'

    # local_dir = f'{DATA_DIR}/{local_results_path}'

    # # Pull predictions from S3
    # mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100)

    # # Process OLMES prediction files to .parqet
    # metrics_local_path     = process_local_folder(local_results_path, file_type='metrics')
    # predictions_local_path = process_local_folder(local_results_path, file_type='predictions')

    predictions_local_path = '/root/ai2/cookbook/olmo-cookbook/.eval_data/olmo2_anneals_predictions.parquet'
    predictions_local_path = '/root/ai2/cookbook/olmo-cookbook/.eval_data/peteish32_predictions.parquet'

    # Run analysis on .parquet
    results_df = run_analysis(predictions_local_path)

if __name__ == '__main__': main()