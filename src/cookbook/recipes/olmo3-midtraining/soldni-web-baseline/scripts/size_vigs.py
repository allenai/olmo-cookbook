#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "boto3",
#     "click",
#     "tqdm",
# ]
# ///

"""
This script figures out how many tokens are in each vigintile for s
"""

from urllib.parse import urlparse
import boto3
import click
from typing import Callable
import re
import tqdm

PREFIX = "s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer"



def get_size_of_prefix(
        prefix: str,
        ext: str = ".npy",
        filter_fn: Callable[[str], bool] | None = None,
) -> dict[str, int]:

    filter_fn = filter_fn or (lambda _: True)
    bucket, prefix = (p := urlparse(prefix)).netloc, p.path.lstrip("/")
    s3 = boto3.client("s3")

    total_prefix_sizes: dict[str, int] = {}
    continuation_token = None

    with tqdm.tqdm(total=None, desc="Getting size of prefix") as pbar:
        while True:
            if continuation_token:
                response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                    ContinuationToken=continuation_token
                )
            else:
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

            for obj in response.get("Contents", []):
                pbar.update(1)

                if "Key" not in obj:
                    continue

                if not obj["Key"].endswith(ext):
                    continue

                if not filter_fn(obj["Key"]):
                    continue

                if "Size" not in obj:
                    continue

                current_prefix = "s3://ai2-llm/" + "/".join(obj["Key"].split("/")[:-1])
                total_prefix_sizes[current_prefix] = total_prefix_sizes.get(current_prefix, 0) + int(obj["Size"])

            pbar.refresh()

            if response.get("IsTruncated", False):
                continuation_token = response.get("NextContinuationToken")
            else:
                break


    return total_prefix_sizes



@click.command()
@click.option("-t", "--token-count", type=int, required=True)
@click.option("-m", "--min-vigintile", type=int, required=True)
def main(token_count: int, min_vigintile: int):
    assert 1 <= min_vigintile <= 20, "min vigintile must be between 1 and 20"
    assert token_count > 0, "token count must be greater than 0"

    # first, we go through all categories and vigintile up to the min vigintile
    def vigintile_filter(key: str, _min_vigintile: int = min_vigintile) -> bool:
        vig_name = re.search(r"vigintile_(\d+)", key)
        if not vig_name:
            return False

        vig_num = int(vig_name.group(1))
        return _min_vigintile <= vig_num

    prefix_sizes = get_size_of_prefix(PREFIX, filter_fn=vigintile_filter)

    current_tokens = sum(prefix_sizes.values()) // 4

    print(f"Current tokens: {current_tokens:,}")
    print(f"Required tokens: {token_count:,}")

    for path in sorted(prefix_sizes.keys()):
        print(f"    - {path}/*.npy")


if __name__ == "__main__":
    main()
