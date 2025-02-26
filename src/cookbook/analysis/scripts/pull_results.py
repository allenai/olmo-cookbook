from cookbook.analysis.process.preprocess import sanity_check
from cookbook.analysis.process.aws import process_local_folder

from cookbook.analysis.process.hf import push_parquet_to_hf
from cookbook.analysis.process.aws import mirror_s3_to_local
from cookbook.analysis.utils import DATA_DIR
from pathlib import Path

def main():
    """
    Mirror AWS bucket to a local folder
    https://us-east-1.console.aws.amazon.com/s3/buckets/ai2-llm?prefix=eval-results/downstream/metaeval/OLMo-ladder/&region=us-east-1&bucketType=general
    """
    bucket_name = "ai2-llm"
    s3_prefix = ["evaluation/anneal-peteish-7b"]
    local_results_path = 'olmo2_anneals_with_pdf'

    local_dir = f'{DATA_DIR}/{local_results_path}'

    mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100)

    sanity_check(local_results_path) # check if any evals are missing

    # Launch preprocessing job!
    metrics_local_path     = process_local_folder(local_results_path, file_type='metrics')
    predictions_local_path = process_local_folder(local_results_path, file_type='predictions')

    # Push to HF!
    push_parquet_to_hf(
        parquet_file_path=metrics_local_path,
        hf_dataset_name='allenai/debug-evals',
        split_name='benchmarks',
        overwrite=True,
        private=True
    )
    push_parquet_to_hf(
        parquet_file_path=predictions_local_path,
        hf_dataset_name='allenai/debug-evals',
        split_name='instances',
        overwrite=True,
        private=True
    )


if __name__ == '__main__':
    main()