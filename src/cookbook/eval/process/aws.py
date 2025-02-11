import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import boto3

# Add parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils import DATA_DIR

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


def mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100, excluded_file_names=EXCLUDED_FILE_NAMES, s3_prefix_list=None):
    """ Recursively download an S3 folder to mirror remote """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    keys = []

    if s3_prefix_list is not None:
        with open(s3_prefix_list, 'r') as file:
            s3_prefixes = [line.strip() for line in file.readlines() if line.strip()]
    else:
        s3_prefixes = [s3_prefix]

    with ThreadPoolExecutor(max_workers=100) as executor:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
        keys = []
        for result in tqdm(executor.map(fetch_page, pages), desc="Listing S3 entries"):
            keys.extend(result)

        for s3_prefix in tqdm(s3_prefixes, desc="Processing S3 prefixes", unit="prefix"):
            pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
            for result in executor.map(fetch_page, pages):
                keys.extend(result)

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

