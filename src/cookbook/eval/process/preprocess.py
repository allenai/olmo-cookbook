import json, os, re, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import psutil
from tqdm import tqdm

# Add parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils import DATA_DIR

# Metrics to use when processing instance-level results
METRICS_TO_KEEP = [
    "acc_raw",
    "acc_per_char",
    "predicted_index_per_char",
    "predicted_index_raw",
    "correct_choice",
    "logits_per_char_corr",
    "logits_per_byte_corr",

    # Generative metrics
    "exact_match",
    "f1",
    "recall",
    "pass_at_1",
    "pass_at_10",

    # Perplexity metrics (e.g., Paloma)
    "bits_per_byte"
]

MODEL_OUTPUT_TO_KEEP = [
    "sum_logits",
    "logits_per_char",
    "logits_per_byte",
]

SIZE_PREFIXES = [
    f'-{size}-' for size in ['3B', '1B', '760M', '750M', '530M', '370M', '300M', '190M', '150M', '90M', '60M', '20M', '4M']
]
SIZE_PREFIXES_FIX = {'3B': '3.2B', '1B': '1.3B'}

CHINHILLA_MULT = [
    '0.5xC', '1xC', '2xC', '5xC', '10xC', '15xC', '20xC'
]

def str_find(str_list, input_string):
    """ Get if a list of strings exists in a string. Return first match """
    hits = [item for item in str_list if item in input_string]
    if len(hits) == 0: 
        return None
    else:
        return hits[0]
    

def get_mix(model_name):
    """ falcon_and_cc_eli5_oh_top10p-3B-5xC => falcon_and_cc_eli5_oh_top10p """
    mix = None
    for prefix in SIZE_PREFIXES:
        if prefix in model_name:
            mix = model_name.split(prefix)[0]
            
            # manual overrides for model ladder
            mix = mix.replace('-rerun', '')
            mix = mix.replace('-moreeval', '-ladder')
    return mix


def extract_step(input_string):
    if input_string is None: return None
    match = re.search(r'step(\d+)(?:-[a-zA-Z0-9]+)?', input_string)
    return int(match.group(1)) if match else None


def remove_prefix(input_string):
    return re.sub(r'^task-\d+-', '', input_string)


def nested_defaultdict():
    return defaultdict(nested_defaultdict)


def fsize(file_path):
    return os.path.getsize(file_path) / (1024 ** 3)


def process_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def get_native_id(pred):
    if pred['native_id'] is not None:
        return str(pred['native_id'])
    elif pred['doc_id'] is not None:
        return str(pred['doc_id'])

def compute_mean_safe(predictions, key):
    values = [
        pred[key] for pred in predictions
        if key in pred and pred[key] is not None
    ]
    return np.array(values).mean().item() if values else None
            

def process_predictions(file_path):
    """ Process a predictions.jsonl to a list """
    predictions = process_jsonl(file_path)

    request_ids_to_bytes = None

    processed = []
    for pred in predictions:
        entry = {}

        entry['native_id'] = get_native_id(pred)
        metrics = pred['metrics']
        model_output = pred['model_output']

        metrics_to_keep = METRICS_TO_KEEP
        model_output_to_keep = MODEL_OUTPUT_TO_KEEP
        
        for col in metrics_to_keep:
            entry[col] = metrics[col] if col in metrics else None
        for col in model_output_to_keep:
            entry[col] = [output[col] if col in output else None for output in model_output]

        # For some generation benchmarks, correct_choice is a str, but this will cause a type error
        # when indexing this column
        if isinstance(entry['correct_choice'], str):
            entry['correct_choice'] = 0

        # Sometimes exact_match is bool when it should be float
        if isinstance(entry['exact_match'], bool):
            entry['exact_match'] = float(entry['exact_match'])

        # Compute BPB using request files
        if request_ids_to_bytes is not None:
            all_num_bytes = request_ids_to_bytes[str(entry["native_id"])]
            if len(all_num_bytes) > len(model_output):
                # For whatever reason ian's results have zero- and few-shot...
                # print(f'Seeing len(entry_requests)={len(entry_requests)} and len(model_output)={len(model_output)}. Truncating...')
                all_num_bytes = all_num_bytes[:len(model_output)]
            # assert len(entry_requests) == len(model_output), (entry_requests, entry["native_id"], requests[0])
            assert len(all_num_bytes) == len(model_output), (len(all_num_bytes), len(model_output))
            all_logits_per_byte = []
            for num_bytes, out in zip(all_num_bytes, model_output):
                LOG_2_OF_E = 1.44269504089
                logits_per_byte = -LOG_2_OF_E * (out["sum_logits"] / num_bytes)
                out['num_bytes'] = num_bytes
                out['logits_per_byte'] = logits_per_byte
                all_logits_per_byte.append(logits_per_byte)
            entry["logits_per_byte"] = all_logits_per_byte
            if 0 <= entry["correct_choice"] < len(all_logits_per_byte):
                entry["logits_per_byte_corr"] = all_logits_per_byte[entry["correct_choice"]]
            else:
                print(f'Incorrect correct_choice indexer: {entry["correct_choice"]}, {file_path}')
                entry["logits_per_byte_corr"] = 0
 
        processed += [entry]
    return processed


def process_metrics(file_path):
    """ Process a metrics.json to a dict """
    with open(file_path, 'r') as f:
        results = json.load(f)

    if 'beaker_info' in results:    del results['beaker_info']
    if 'compute_config' in results: del results['compute_config']
    if 'task_config' in results:    del results['task_config']

    # Only keep these metrics for Paloma
    PALOMA_METRICS = [
        'bits_per_byte',
        'ppl_token',
        'ppl_char',
        'ppl_word',
        'ppl_byte',
    ]

    if 'metrics' in results:
        for metric in results['metrics']:
            if ('paloma' in file_path or 'llm_compression' in file_path or 'custom_loss' in file_path) and metric not in PALOMA_METRICS:
                continue
            results[metric] = results['metrics'][metric]

    # Get token spend if it exists (num_instances is already a col)
    if 'extra_metrics' in results and 'num_tokens' in results["extra_metrics"]:
        results["num_tokens"] = results['extra_metrics']["num_tokens"]

    # Rename bpb to logits_per_byte_corr if it exists
    if 'bits_per_byte' in results and results['bits_per_byte'] is not None:
        results['logits_per_byte_corr'] = results['bits_per_byte']

    if 'logits_per_byte_corr' not in results:
        # Get bits-per-byte from prediction files if they dont exist
        predictions_path = file_path.replace('metrics.json', 'predictions.jsonl')
        if os.path.exists(predictions_path):
            predictions = process_predictions(predictions_path)

            for prediction in predictions:
                if 'correct_choice' in prediction and prediction['correct_choice'] is not None:
                    try:
                        correct_choice = prediction['correct_choice']

                        if ('logits_per_byte_corr' not in prediction or prediction['logits_per_byte_corr'] is None) and 'logits_per_byte' in prediction:
                            logits_per_byte = prediction['logits_per_byte']

                            if 0 <= correct_choice < len(logits_per_byte):
                                prediction['logits_per_byte_corr'] = logits_per_byte[correct_choice]
                            else:
                                # print(f'Incorrect correct_choice indexer: {correct_choice}, {file_path}')
                                prediction['logits_per_byte_corr'] = 0

                        if ('logits_per_char_corr' not in prediction or prediction['logits_per_char_corr'] is None) and 'logits_per_char' in prediction:
                            logits_per_char = prediction['logits_per_char']

                            if 0 <= correct_choice < len(logits_per_char):
                                prediction['logits_per_char_corr'] = logits_per_char[correct_choice]
                            else:
                                # print(f'Incorrect correct_choice indexer: {correct_choice}, {file_path}')
                                prediction['logits_per_char_corr'] = 0
                    except Exception as e:
                        print(e)
                        raise RuntimeError(prediction, results)

            logits_per_byte = compute_mean_safe(predictions, 'logits_per_byte_corr')
            logits_per_char = compute_mean_safe(predictions, 'logits_per_char_corr')

            if 'logits_per_byte_corr' not in results: 
                results['logits_per_byte_corr'] = logits_per_byte
            if 'logits_per_char_corr' not in results: 
                results['logits_per_char_corr'] = logits_per_char

    return results


def process_chunk(chunk):
    return pd.DataFrame(chunk)


def get_available_cpus(threshold=80):
    cpu_usages = psutil.cpu_percent(percpu=True)
    available_cpus = [i for i, usage in enumerate(cpu_usages) if usage < threshold]
    return available_cpus


def load_df_parallel(data, file_type, usage_threshold=80):
    """ Load data as df w/ a CPU pool. Only use CPUs with usage below usage_threshold """
    available_cpus = get_available_cpus(threshold=usage_threshold)

    if file_type == 'metrics':
        num_partitions = max(1, len(data) // 1_000)
    elif file_type == 'predictions':
        num_partitions = max(1, len(data) // 100_000) # default is 10_000, 50K chunks led to a broken pipe

    print(f'Distributing {num_partitions} chunks across {len(available_cpus)} CPUs')
    
    if num_partitions == 0:
        raise RuntimeError("No CPUs are available below the usage threshold.")
    
    # Use numpy for efficient chunking
    chunk_size = len(data) // num_partitions
    remainder = len(data) % num_partitions
    chunks = [data[i * chunk_size + min(i, remainder) : (i + 1) * chunk_size + min(i + 1, remainder)] for i in range(num_partitions)]

    print('Launching parallel processing...')
    
    with Pool(processes=len(available_cpus)) as pool:
        dataframes = list(tqdm(pool.imap(process_chunk, chunks), desc='Converting to Pandas dataframe', total=len(chunks)))

    # with Pool(processes=num_partitions) as pool:
    #     dataframes = list(tqdm(pool.map(process_chunk, chunks), desc='Converting to Pandas dataframe', total=len(chunks)))
    
    return pd.concat(dataframes, ignore_index=True)


def get_metadata_from_file_name(root, file):
    path_parts = os.path.normpath(root).split(os.sep)

    # Use last two folders as "path"/"step":
    # E.g., ../peteish-moreeval-1B-0.5xC/step8145-unsharded-hf
    if len(path_parts) >= 2:
        if 'step' in path_parts[-1]: # and ('-unsharded' in path_parts[-1] or '-hf' in path_parts[-1])
            # Local OLMo runs (anything that ends in "stepXXX-unsharded")
            model_name = path_parts[-2]
            step_str = path_parts[-1]
        else:
            # External models (e.g., llama)
            model_name = path_parts[-1]
            step_str = None
    else:
        raise RuntimeError(f'Could not process path: {path_parts}')

    # Get task name: "arc_challenge-metrics.json" => "arc_challenge"
    task = remove_prefix(file) # Remove "task-XXX" prefix: task-XXX-task_name.json => task_name.json
    task = task.rsplit('-', 1)[0]

    # Get step name: "stepXXX-unsharded" => "XXX"
    step = extract_step(step_str)

    # Get mix name
    mix_name = get_mix(model_name)

    # Get other metadata
    size = str_find(SIZE_PREFIXES, model_name)
    if size is not None: size = size.replace('-', '')
    token_ratio = str_find(CHINHILLA_MULT, model_name)

    return model_name, mix_name, step, step_str, size, token_ratio, task


def load_file(file_data, _type):
    root, file = file_data
    file_path = os.path.join(root, file)

    model_name, mix_name, step, step_str, size, token_ratio, task = get_metadata_from_file_name(root, file)

    if _type == 'predictions':
        # Load predictions
        if 'predictions.jsonl' not in file_path: 
            return []
        results = process_predictions(file_path)
    elif _type == 'metrics':
        if 'metrics.json' not in file_path:
            return []
        if 'verbose-metrics.json' in file_path:
            return []
        metrics = process_metrics(file_path)

        # Sometimes the metrics file causes OOM errors, so we will delete if it's too big
        if 'metrics' in metrics and len(str(metrics['metrics'])) > 1000:
            metrics['metrics'] = None
        
        results = [metrics]

    # Add metadata to parquet file
    for result in results:
        result.update({
            'model': model_name,
            'mix': mix_name,
            'step': step,
            'size': size,
            'token_ratio': token_ratio,
            'step_str': step_str,
            'task': task,
            's3_path': file_path,
        })
    
    return results


def process_files_chunk(files_chunk, _type):
    results = []
    for file in files_chunk:
        results.extend(load_file(file, _type))
    return results


def scan_dir(data_input):
    all_files = []

    # Ensure the input is a list of paths, even if it's a single path
    paths = [data_input] if isinstance(data_input, (str, os.PathLike)) else data_input
    if not isinstance(paths, (list, tuple)):
        raise ValueError("Input must be a directory path or a list of paths.")

    with tqdm(desc="Scanning paths", total=len(paths), unit="path") as pbar:
        for path in paths:
            if os.path.isfile(path):
                if path.endswith('-predictions.jsonl') or path.endswith('-metrics.json'):
                    all_files.append((os.path.dirname(path), os.path.basename(path)))
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    if 'local_testing' not in root:
                        all_files.extend(
                            (root, file)
                            for file in files
                            if file.endswith('-predictions.jsonl') or file.endswith('-metrics.json')
                        )
            pbar.update(1)
    
    return all_files


def recursive_pull(data_dir, file_type):
    all_files = scan_dir(data_dir)

    # all_files = all_files[:10_000] # for testing
    # all_files = all_files[:1_000_000] # for testing

    # chunk_size = 100 # for testing
    chunk_size = 1_000

    all_preprocessed = []
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    total_files = len(all_files)

    with tqdm(total=len(file_chunks), desc="Submitting file chunks") as submit_pbar:
        with tqdm(total=total_files, desc=f"Recursively loading files in {data_dir.name}") as pbar:
            with ProcessPoolExecutor(max_workers=len(get_available_cpus())) as executor:
                futures = {}
                for chunk in file_chunks:
                    future = executor.submit(process_files_chunk, chunk, file_type)
                    futures[future] = len(chunk)
                    submit_pbar.update(1)  # Update submission progress
                for future in as_completed(futures):
                    all_preprocessed.extend(future.result())
                    pbar.update(futures[future])  # Update based on the chunk size
            pbar.close()
        submit_pbar.close()

    return all_preprocessed


def cleanup_metrics_df(df):
    """ A safe function to clean up benchmark results """
    # Preprocess the df into a usuable format
    df = df.drop(columns=["Unnamed: 0"], errors='ignore')

    # Modify column order to move these up
    desired_order = ['task', 'model', 'step', 'mix', 'size', 'token_ratio', 'primary_score', 'logits_per_byte_corr']
    existing_columns = [col for col in desired_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_columns]
    df = df[existing_columns + remaining_cols]

    # Add primary score if it does not exist
    if 'primary_score' in df.columns:
        if 'acc_per_char' in df.columns:
            df['primary_score'] = df['primary_score'].fillna(df['acc_per_char'])
        if 'exact_match' in df.columns:
            df['primary_score'] = df['primary_score'].fillna(df['exact_match'])
        if 'pass_at_1' in df.columns:
            df['primary_score'] = df['primary_score'].fillna(df['pass_at_1'])

    return df


def verify_df(df):
    # Identify missing models/tasks
    unique_models = df['model'].unique()
    unique_tasks  = df['task'].unique()
    missing_entries = []
    for model in unique_models:
        for task in unique_tasks:
            task_rows = df[(df['model'] == model) & (df['task'] == task)]
            if task_rows.empty:
                missing_entries.append((model, task))

    if missing_entries:
        print("Missing tasks for models:")
        for model, task in missing_entries:
            print(f"  - Model: {model}, Task: {task}")


def sanity_check(folder_name):
    """ 
    All leaf folders should have the same eval data. This prints folders
    that do not have data compared to evals that appear at least once.
    """
    data_dir = Path(DATA_DIR).resolve()
    data_dir.mkdir(exist_ok=True)

    aws_dir = data_dir / folder_name

    all_files = scan_dir(aws_dir)

    model_tasks = defaultdict(set)
    for (root, file) in all_files:
        model_name, mix_name, step, step_str, size, token_ratio, task = get_metadata_from_file_name(root, file)
        # synthetic evals (excluded from Ian's model for now)
        if ':cot' in task:
            continue
        if ':pertub_cot' in task:
            continue
        if ':para' in task:
            continue
        if ':perturb_cot' in task:
            continue
        if ':distractors' in task:
            continue
        if ':enlarge' in task:
            continue
        if ':perturb_rc' in task:
            continue
        if 'bbh_' in task:
            continue

        # only math/code
        if not ('gsm8k' in task or 'mbpp' in task or 'codex' in task or 'minerva' in task):
            continue

        # perplexity evals
        if '-verbose' in task:
            continue
        if task in ["paloma_twitterAAE_HELM_fixed", "paloma_c4_100_domains", "paloma_dolma_100_subreddits"]:
            # these tasks are half-evaluated and shouldn't be in there anyways
            continue
        # if 'paloma' in task or 'llm_compression' in task or 'custom_loss' in task:
        #     continue
        model_tasks[f'{model_name}-{step}'].add(task)
        # model_tasks[f'{model_name}'].add(task)

    all_tasks = set(task for tasks in model_tasks.values() for task in tasks)
    incomplete_models = {model for model, tasks in model_tasks.items() if tasks != all_tasks}
    for model in incomplete_models:
        # print(f"Model {model} is missing tasks. Missing tasks: {all_tasks - model_tasks[model]}")
        print(f"({model}, {list(all_tasks - model_tasks[model])})")
