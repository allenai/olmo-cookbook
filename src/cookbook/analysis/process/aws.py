import sys, os, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import boto3

from cookbook.analysis.process.preprocess import fsize, recursive_pull, load_df_parallel, cleanup_metrics_df
from cookbook.analysis.utils import DATA_DIR


# Add parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

EXCLUDED_FILE_NAMES = [
    'requests.jsonl',
    'recorded-inputs.jsonl',
    'metrics-all.jsonl'
]


def download_file(s3_client, bucket_name, key, local_dir, excluded_file_names):
    local_path = os.path.join(local_dir, key)

    if any(f in key.split('/')[-1] for f in excluded_file_names):
        return # Skip download if there are any str matches with EXCLUDED_FILE_NAMES
    
    if os.path.exists(local_path):
        s3_head = s3_client.head_object(Bucket=bucket_name, Key=key)
        s3_file_size = s3_head['ContentLength']
        local_file_size = os.path.getsize(local_path)
        if s3_file_size == local_file_size:
            return  # Skip download if the file already exists and has the same size
        
    # 1) Remove any subfolders after checkpoint: [MODEL_NAME]/step2693-hf/gsm8k__olmes/result.json => step2693-hf/result.json
    local_path = re.sub(r'([^/]+)/[^/]+/([^/]+)$', r'\1/\2', local_path)
    
    # 2) Remove any subfolders after checkpoint: [MODEL_NAME]/stepXXXX???/.../... => stepXXXX???/file
    local_path = re.sub(r'(step\d+[^/]*)/.*?/([^/]+)$', r'\1/\2', local_path)

    # 3) Remove the task-XXX- prefix
    local_path = re.sub(r'task-\d+-', '', local_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket_name, key, local_path)


def fetch_page(page):
    return [obj['Key'] for obj in page.get('Contents', [])]


def fetch_keys_for_prefix(bucket_name, prefix):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    keys = []
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    for page in pages:
        keys.extend(fetch_page(page))
    return keys


def mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100, excluded_file_names=EXCLUDED_FILE_NAMES):
    """ Recursively download an S3 folder to mirror remote """
    s3_client = boto3.client('s3')
    keys = []

    if isinstance(s3_prefix, tuple):
        s3_prefixes = list(s3_prefix)
    elif not isinstance(s3_prefix, list):
        s3_prefixes = [s3_prefix]
    else:
        s3_prefixes = s3_prefix

    print(f'Searching through S3 prefixes: {s3_prefixes}')

    # with ProcessPoolExecutor() as executor:
    with ThreadPoolExecutor(max_workers=100) as executor:
        future_to_prefix = {}
        with tqdm(total=len(s3_prefixes), desc="Submitting S3 prefix tasks", unit="prefix") as submit_pbar:
            for prefix in s3_prefixes:
                future = executor.submit(fetch_keys_for_prefix, bucket_name, prefix)
                future_to_prefix[future] = prefix
                submit_pbar.update(1)

        with tqdm(total=len(future_to_prefix), desc="Fetching keys from S3 prefixes", unit="prefix") as pbar:
            for future in as_completed(future_to_prefix):
                try:
                    keys.extend(future.result())
                except Exception as e:
                    print(f"Error processing prefix {future_to_prefix[future]}: {e}")
                pbar.update(1)

    if max_threads > 1:
        # ProcessPoolExecutor seems not to work with AWS, so we use ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(download_file, s3_client, bucket_name, key, local_dir, excluded_file_names): key for key in keys}
            with tqdm(total=len(futures), desc="Syncing download folder from S3", unit="file") as pbar:
                for _ in as_completed(futures):
                    pbar.update(1)
    else:
        # Sequential implmentation
        for key in tqdm(keys, desc="Syncing download folder from S3", unit="file"):
            download_file(s3_client, bucket_name, key, local_dir, excluded_file_names)


def process_local_folder(local_results_path, file_type='predictions'):
    data_dir = Path(DATA_DIR).resolve()
    data_dir.mkdir(exist_ok=True)

    aws_dir         = data_dir / local_results_path
    prediction_path = data_dir / f"{local_results_path}_predictions.parquet"
    metrics_path    = data_dir / f"{local_results_path}_metrics.parquet"

    predictions_df = recursive_pull(aws_dir, file_type)

    # Save predictions to parquet
    import time
    start_time = time.time()
    
    df = load_df_parallel(predictions_df, file_type) # for 6700 preds: 300s (5 min)

    print(f"Converted to pandas in: {time.time() - start_time:.4f} seconds")

    if file_type == 'metrics':
        df = cleanup_metrics_df(df)

        print(df.columns)

        df.to_parquet(metrics_path)
        print('Done!')
        return

    # Reset the df index (for faster indexing)
    df.set_index(['task', 'model', 'step', 'mix'], inplace=True)

    # Save to parquet
    df.to_parquet(prediction_path, index=True)
    print(f"Predictions saved to {prediction_path} ({fsize(prediction_path):.2f} GB)")

    print('Done!')

    return prediction_path