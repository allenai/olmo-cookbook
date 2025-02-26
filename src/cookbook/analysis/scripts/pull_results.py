from cookbook.analysis.process.preprocess import fsize, recursive_pull, load_df_parallel, cleanup_metrics_df, sanity_check
from cookbook.analysis.process.hf import push_parquet_to_hf
from cookbook.analysis.process.aws import mirror_s3_to_local
from cookbook.analysis.utils import DATA_DIR
from pathlib import Path


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