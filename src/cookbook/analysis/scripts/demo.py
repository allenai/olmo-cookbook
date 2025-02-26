from cookbook.analysis.process.hf import pull_predictions_from_hf
from cookbook.analysis.process.aws import mirror_s3_to_local, process_local_folder
from cookbook.analysis.utils import DATA_DIR
from cookbook.analysis.run_analysis import run_analysis


def main():
    # local_path = pull_predictions_from_hf("allenai/ladder-evals", "benchmarks")
    # df.loc[df['mix'] == 'baseline', 'mix'] = 'dolma17' # # fix for the names of one of Ian's data mixes
    # MODELS = list(df['model'].unique())
    # TASKS = sorted(list(df['task'].unique()))

    # Pull pre-processed predictions from HF
    predictions_local_path = pull_predictions_from_hf("allenai/peteish32-evals", "instances")
    predictions_local_path = pull_predictions_from_hf("allenai/olmo2-anneals-evals", "main")

    # Pull predictions from AWS
    bucket_name = "ai2-llm"
    s3_prefix = ["evaluation/anneal-peteish-7b"]
    local_results_path = 'olmo2_anneals'

    bucket_name = 'ai2-llm'
    s3_prefix = ['evaluation/peteish32/']
    local_results_path = 'peteish32'

    local_dir = f'{DATA_DIR}/{local_results_path}'

    # Pull predictions from S3
    mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100)

    # Process OLMES prediction files to .parqet
    metrics_local_path     = process_local_folder(local_results_path, file_type='metrics')
    predictions_local_path = process_local_folder(local_results_path, file_type='predictions')

    # predictions_local_path = '/root/ai2/cookbook/olmo-cookbook/.eval_data/olmo2_anneals_predictions.parquet'
    predictions_local_path = '/root/ai2/cookbook/olmo-cookbook/.eval_data/peteish32_predictions.parquet'

    # Run analysis on .parquet
    results_df = run_analysis(predictions_local_path)


if __name__ == '__main__': main()