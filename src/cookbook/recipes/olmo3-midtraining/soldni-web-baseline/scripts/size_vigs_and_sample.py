#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "boto3",
#     "click",
#     "tqdm",
#     "pyyaml",
# ]
# ///

"""
This script figures out how many tokens are in each vigintile
"""

import random
from urllib.parse import urlparse
import boto3
import click
from typing import Callable
import re
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import yaml

NPY_PREFIX = "s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer"
DATA_PREFIX = "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3/weborganizer_ft/dclm_plus2_vigintiles"


def process_objects_batch(objects: list, ext: str, filter_fn: Callable[[str], bool], lock: Lock, total_file_sizes: dict[str, int], pbar: tqdm.tqdm) -> None:
    local_sizes: dict[str, int] = {}

    for obj in objects:
        if "Key" not in obj:
            continue

        if not obj["Key"].endswith(ext):
            continue

        if not filter_fn(obj["Key"]):
            continue

        if "Size" not in obj:
            continue

        file_path = "s3://ai2-llm/" + obj["Key"]
        local_sizes[file_path] = int(obj["Size"])

    with lock:
        for file_path, size in local_sizes.items():
            total_file_sizes[file_path] = size
        pbar.update(len(objects))


def get_size_of_prefix(
        prefix: str,
        ext: str = ".npy",
        filter_fn: Callable[[str], bool] | None = None,
        max_workers: int = 10,
        batch_size: int = 1000,
) -> dict[str, int]:

    filter_fn = filter_fn or (lambda _: True)
    bucket, prefix = (p := urlparse(prefix)).netloc, p.path.lstrip("/")
    s3 = boto3.client("s3")

    total_file_sizes: dict[str, int] = {}
    lock = Lock()
    continuation_token = None
    all_objects = []

    with tqdm.tqdm(total=None, desc="Listing objects") as list_pbar:
        while True:
            if continuation_token:
                response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                    ContinuationToken=continuation_token
                )
            else:
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

            objects = response.get("Contents", [])
            all_objects.extend(objects)
            list_pbar.update(len(objects))

            if response.get("IsTruncated", False):
                continuation_token = response.get("NextContinuationToken")
            else:
                break

    with tqdm.tqdm(total=len(all_objects), desc="Processing objects") as process_pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for i in range(0, len(all_objects), batch_size):
                batch = all_objects[i:i + batch_size]
                future = executor.submit(
                    process_objects_batch,
                    batch,
                    ext,
                    filter_fn,
                    lock,
                    total_file_sizes,
                    process_pbar
                )
                futures.append(future)

            for future in as_completed(futures):
                future.result()

    return total_file_sizes


@click.command()
@click.option("-t", "--token-count", type=int, required=True)
@click.option("-m", "--min-vigintile", type=int, required=True)
@click.option("-n", "--npy-prefix", type=str, default=NPY_PREFIX)
@click.option("-d", "--data-prefix", type=str, default=DATA_PREFIX)
@click.option("-o", "--output", type=str, required=True)
def main(token_count: int, min_vigintile: int, npy_prefix: str, data_prefix: str, output) -> None:

    random.seed(42)

    assert 1 <= min_vigintile <= 20, "min vigintile must be between 1 and 20"
    assert token_count > 0, "token count must be greater than 0"

    # first, we go through all categories and vigintile up to the min vigintile
    def vigintile_filter(key: str, _min_vigintile: int = min_vigintile) -> bool:
        vig_name = re.search(r"vigintile_(\d+)", key)
        if not vig_name:
            return False

        vig_num = int(vig_name.group(1))
        return _min_vigintile <= vig_num

    print(f"Sizing npy files in {npy_prefix}...")
    npy_prefix_sizes = get_size_of_prefix(npy_prefix, ext=".npy", filter_fn=vigintile_filter)

    print(f"Sizing data files in {data_prefix}...")
    data_prefix_sizes = get_size_of_prefix(data_prefix, ext='.zst', filter_fn=vigintile_filter)

    actual_token_size = sum(npy_prefix_sizes.values()) // 4
    print(f"Actual token size: {actual_token_size:,} tokens")

    sampling_ratio = token_count / actual_token_size
    print(f"Sampling ratio: {sampling_ratio:.4f}")

    desired_file_total = sampling_ratio * sum(data_prefix_sizes.values())
    print(f"Desired file total: {desired_file_total / 1024 ** 3:.2f} GB")

    topic_vig_hier = {}
    for key, size in data_prefix_sizes.items():
        *_, topic, vigintile, _ = key.split("/")
        topic = topic.replace("topic_", "")
        vigintile = int(vigintile.replace("vigintile_", ""))
        topic_vig_hier.setdefault(topic, {}).setdefault(vigintile, {})[key] = size

    destination = npy_prefix\
        .replace("s3://", "/mnt/raid0/")\
        .replace("/allenai/dolma2-tokenizer", f"_{token_count // 1024 ** 3}B/allenai/dolma2-tokenizer")

    output_obj = {
        "documents": [],
        "destination": destination,
        "tokenizer": {
            "name_or_path": "allenai/dolma2-tokenizer",
            "eos_token_id": 100257,
            "pad_token_id": 100277,
            "encode_special_tokens": True,
        },
        "ring_size": 8,
        "processes": 128,
        "max_size": 4_000_000_000,
        "sample_ring_prop": True,
        "dtype": "uint32",
    }
    with tqdm.tqdm(total=desired_file_total, desc="Selecting files", unit_scale=True) as pbar:
        for topic, vigintiles in topic_vig_hier.items():
            for vigintile, files in vigintiles.items():
                current_file_size = sum(files.values())
                desired_file_total = sampling_ratio * current_file_size
                current_files = sorted(files.items())
                random.shuffle(current_files)

                current_size = 0
                for file_path, size in current_files:
                    if current_size > desired_file_total:
                        break

                    output_obj["documents"].append(file_path)
                    current_size += size
                    pbar.update(size)


    output_obj["documents"] = sorted(output_obj["documents"])

    with open(output, "w") as f:
        yaml.safe_dump(output_obj, f)


if __name__ == "__main__":
    main()
