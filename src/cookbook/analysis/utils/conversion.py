from collections import defaultdict
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import json
from multiprocessing import Pool, cpu_count
import time
from typing import List
from cookbook.eval.cache import get_datalake_cache
from cookbook.eval.datalake import MetricsAll, Prediction

from tqdm import tqdm


def get_schema():
    """Instance-level prediction table schema"""
    return {
        'alias': str,
        'model_name': str,
        'model_path': str,
        'revision': str,
        'task_name': str,
        'doc_id': int,
        'native_id': str,
        'label': str,
        'num_tokens': int,
        'correct_choice': int,
        'bits_per_byte_corr': float,
        'primary_score': float,
        'pass_at_1': float,
        'pass_at_10': float,
        'exact_match': float,
        'f1': float,
        'recall': float,
        # 'logits_per_char': List[float],
        # 'bits_per_byte': List[float],
        # 'logits_per_char_corr': float,
        # 'logits_per_byte_corr': float,
    }


def _truncate_for_display(value, max_length=500):
    """
    Truncate a value for display in error messages.
    """
    if value is None:
        return "None"
    
    str_value = str(value)
    if len(str_value) <= max_length:
        return str_value
    else:
        return str_value[:max_length] + "..."


def _enforce_schema(row, schema):
    """Enforce the exact schema. Uses NaN for missing data"""
    def get_default_value(expected_type):
        """Get default value for a type when column is missing or conversion fails."""
        if expected_type == str:
            return ""
        elif expected_type in (int, float):
            return np.nan
        elif expected_type == List[float]:
            return []
        else:
            return None
    
    def convert_value(value, expected_type):
        """Convert value to expected type with proper null handling."""
        if value is None or pd.isna(value):
            return get_default_value(expected_type)
        
        if expected_type == str:
            return str(value)
        elif expected_type == int:
            return int(float(value)) if not isinstance(value, bool) else int(value)
        elif expected_type == float:
            return float(value)
        elif expected_type == List[float]:
            if isinstance(value, list):
                # Ensure all elements are floats
                try:
                    return [float(x) for x in value if x is not None and not pd.isna(x)]
                except (ValueError, TypeError):
                    return []
            elif isinstance(value, str):
                # Handle case where it might be a string representation
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [float(x) for x in parsed if x is not None and not pd.isna(x)]
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
                return []
            else:
                # Single value - wrap in list
                try:
                    return [float(value)] if value is not None and not pd.isna(value) else []
                except (ValueError, TypeError):
                    return []
        else:
            return value
    
    enforced_row = {}
    
    for col_name, expected_type in schema.items():
        if col_name in row:
            try:
                enforced_row[col_name] = convert_value(row[col_name], expected_type)
            except (ValueError, TypeError, OverflowError):
                enforced_row[col_name] = get_default_value(expected_type)
        else:
            enforced_row[col_name] = get_default_value(expected_type)
    
    return enforced_row


def _process_prediction_row(metrics, prediction, model_output_dict):
    """
    Process a single prediction row with common data cleaning logic.
    Returns the processed row ready for schema enforcement.
    """
    # Build the full row first
    full_row = {
        "alias": metrics.alias,
        "model_name": metrics.model_name,
        "model_path": metrics.model_path,
        **metrics.to_dict(),
        **prediction.to_dict(),
        **model_output_dict,
    }

    # Expand metrics dict but don't override existing columns
    for metric_key, metric_value in full_row["metrics"].items():
        if metric_key not in full_row:
            full_row[metric_key] = metric_value

    # Apply data cleaning for schema compatibility
    if "correct_choice" not in full_row or isinstance(full_row["correct_choice"], str):
        full_row["correct_choice"] = 0

    if "exact_match" in full_row and isinstance(full_row["exact_match"], bool):
        full_row["exact_match"] = float(full_row["exact_match"])

    if not isinstance(full_row["native_id"], str):
        full_row["native_id"] = str(full_row["native_id"])

    if not isinstance(full_row["label"], str):
        full_row["label"] = str(full_row["label"])

    # Fix legacy names
    if "bits_per_byte_corr" in full_row and full_row["bits_per_byte_corr"] is not None:
        full_row["logits_per_byte_corr"] = full_row["bits_per_byte_corr"]

    if "logits_per_byte_corr" in full_row and full_row["logits_per_byte_corr"] is not None:
        full_row["bits_per_byte_corr"] = full_row["logits_per_byte_corr"]

    # Get the instance-level primary_metric
    primary_metric = full_row["task_config"]["primary_metric"]
    correct_choice = full_row["correct_choice"]
    if primary_metric not in full_row:
        # Unfortunately, not all oe-eval tasks specify a metric, so default to these
        if "acc_per_char" in full_row:
            primary_score = full_row["acc_per_char"]
        elif "exact_match" in full_row:
            primary_score = full_row["exact_match"]
        elif "pass_at_1" in full_row:
            primary_score = full_row["pass_at_1"]
        else:
            raise ValueError("Could not get primary metric for entry: " + _truncate_for_display(full_row) + f" with keys: {full_row.keys()} and with metrics: {full_row['metrics']}")
        assert not isinstance(primary_score, list), _truncate_for_display(full_row)
        full_row["primary_score"] = primary_score
    elif isinstance(full_row[primary_metric], list):
        full_row["primary_score"] = full_row[primary_metric][correct_choice]
    else:
        full_row["primary_score"] = full_row[primary_metric]

    return full_row


def _get_type_mapping():
    """
    Get a mapping of field names to their required PyArrow types based on the enforced schema.
    """
    import pyarrow as pa
    
    schema = get_schema()
    type_mapping = {}
    
    for field_name, python_type in schema.items():
        if python_type == str:
            type_mapping[field_name] = pa.string()
        elif python_type == int:
            # Use Int64 to support null values
            type_mapping[field_name] = pa.int64()
        elif python_type == float:
            type_mapping[field_name] = pa.float64()
        elif python_type == List[float]:
            # Use list of float64 for List[float] fields
            type_mapping[field_name] = pa.list_(pa.float64())
        else:
            # Default to string for unknown types
            type_mapping[field_name] = pa.string()
    
    return type_mapping


def _create_consistent_table(chunk_df, worker_id=None):
    """
    Create a PyArrow table with consistent types for the enforced schema.
    """
    import pyarrow as pa
    
    type_mapping = _get_type_mapping()
    
    # Convert DataFrame to PyArrow table
    table = pa.Table.from_pandas(chunk_df)
    
    # Use enforced schema cating
    schema_fields = []
    for field in table.schema:
        field_name = field.name
        if field_name in type_mapping:
            # Use our enforced type
            target_type = type_mapping[field_name]
            schema_fields.append(pa.field(field_name, target_type))
        else:
            # Keep original type for unknown fields (shouldn't happen with enforced schema)
            schema_fields.append(field)
    
    # Create new schema and cast table
    consistent_schema = pa.schema(schema_fields)
    consistent_table = table.cast(consistent_schema)

    return consistent_table


def _cleanup_dataframe_for_pyarrow(df, worker_id=None):
    """
    Clean up DataFrame to make it compatible with PyArrow conversion using the enforced schema.
    """
    import json
    
    schema = get_schema()
    cleaned_df = df.copy()
    
    for col_name, expected_type in schema.items():
        if col_name in cleaned_df.columns:
            col_data = cleaned_df[col_name]
            
            try:
                if expected_type == str:
                    cleaned_df[col_name] = col_data.apply(
                        lambda x: str(x) if x is not None else ""
                    )
                elif expected_type == int:
                    if col_data.dtype == 'bool':
                        cleaned_df[col_name] = col_data.astype('int64')
                    elif col_data.dtype == 'object':
                        def safe_int_convert(x):
                            if x is None or pd.isna(x):
                                return np.nan
                            if isinstance(x, bool):
                                return int(x)
                            try:
                                return int(float(x))
                            except (ValueError, TypeError):
                                return np.nan
                        cleaned_df[col_name] = col_data.apply(safe_int_convert)
                        # Convert to Int64 dtype to support NaN values
                        cleaned_df[col_name] = cleaned_df[col_name].astype('Int64')
                elif expected_type == float:
                    if col_data.dtype == 'bool':
                        cleaned_df[col_name] = col_data.astype('float64')
                    elif col_data.dtype == 'object':
                        def safe_float_convert(x):
                            if x is None or pd.isna(x):
                                return np.nan
                            if isinstance(x, bool):
                                return float(x)
                            try:
                                return float(x)
                            except (ValueError, TypeError):
                                return np.nan
                        cleaned_df[col_name] = col_data.apply(safe_float_convert).astype('float64')
                elif expected_type == List[float]:
                    # Handle List[float] fields - ensure they are properly formatted lists
                    def safe_list_float_convert(x):
                        if x is None or pd.isna(x):
                            return []
                        if isinstance(x, list):
                            # Clean the list to ensure no None values
                            cleaned_list = []
                            for item in x:
                                if item is not None and not pd.isna(item):
                                    try:
                                        cleaned_list.append(float(item))
                                    except (ValueError, TypeError):
                                        continue
                            return cleaned_list
                        else:
                            # Convert single value to list
                            try:
                                return [float(x)] if not pd.isna(x) else []
                            except (ValueError, TypeError):
                                return []
                    
                    cleaned_df[col_name] = col_data.apply(safe_list_float_convert)
            except Exception as e:
                print(f"[Worker {worker_id}] ⚠️  Failed to clean column {col_name}: {e}")
                # Keep original column if cleaning fails
                continue
    
    return cleaned_df


def _process_experiment_worker(args):
    """
    Worker function for processing a single experiment in parallel with enforced schema.
    """
    experiment, temp_dir, chunk_size, force, skip_on_fail, worker_id = args
    
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    schema = get_schema()
    batch_files = []
    current_chunk = []
    total_processed = 0
    
    try:
        # Get task ids using metrics files for experiment
        cache = get_datalake_cache()
        metrics_list = MetricsAll.run(experiment_id=experiment.experiment_id, force=force, skip_on_fail=skip_on_fail)
        
        for metric in metrics_list:
            try:
                # Get predictions
                if not (result := cache.get(experiment_id=experiment.experiment_id, type="predictions")).success or force:
                    response = requests.get(
                        f"https://oe-eval-datalake.allen.ai/greenlake/download-result/{experiment.experiment_id}",
                        params={"resulttype": "PREDICTIONS", "task_idx": metric.task_idx},
                        headers={"accept": "application/json"},
                    )
                    response.raise_for_status()
                    
                    # Response is a List[dict]
                    parsed = [json.loads(line) for line in response.iter_lines(decode_unicode=True) if line]
                    result = cache.set(parsed, experiment_id=experiment.experiment_id, type="predictions")
                
                # Process predictions
                for prediction_data in (result.value or []):
                    prediction = Prediction(**prediction_data)
                    
                    # Convert list of dicts to dict of lists for model outputs
                    model_output_dict = defaultdict(list)
                    for output in prediction.model_output:
                        for key, value in output.items():
                            model_output_dict[key].append(value)

                    # Process prediction files
                    full_row = _process_prediction_row(metric, prediction, model_output_dict)
                    
                    # Enforce the schema
                    row = _enforce_schema(full_row, schema)
                    current_chunk.append(row)
                    total_processed += 1
                    
                    # Write chunk when it reaches target size
                    if len(current_chunk) >= chunk_size:
                        batch_file = temp_dir / f"worker_{worker_id}_batch_{len(batch_files):06d}.parquet"
                        
                        # Convert to PyArrow table with consistent types
                        chunk_df = pd.DataFrame(current_chunk)
                        
                        # Create table with consistent types for problematic fields
                        table = _create_consistent_table(chunk_df, worker_id=worker_id)
                        
                        pq.write_table(table, batch_file)
                        
                        batch_files.append(batch_file)
                        current_chunk = []
            
            except Exception as e:
                if skip_on_fail:
                    if "404 client error" in str(e).lower():
                        # Some prediction files simply don't exist in the datalake
                        continue
                    print(f"[Worker {worker_id}] ⚠️  Task {metric.task_idx} in experiment {experiment.experiment_id} failed (skipping): {e}")
                    continue
                else:
                    raise e
        
        # Write final chunk if any remaining data
        if current_chunk:
            batch_file = temp_dir / f"worker_{worker_id}_batch_{len(batch_files):06d}.parquet"
            
            # Convert to PyArrow table with consistent types
            chunk_df = pd.DataFrame(current_chunk)
            
            # Create table with consistent types for problematic fields
            table = _create_consistent_table(chunk_df, worker_id=worker_id)
            
            pq.write_table(table, batch_file)
            
            batch_files.append(batch_file)
        
        return batch_files, total_processed, experiment.experiment_id
        
    except Exception as e:
        if skip_on_fail:
            print(f"[Worker {worker_id}] ❌ Experiment {experiment.experiment_id} failed (Processed {total_processed} rows before failure): {e}")

            # Return any batch files that were successfully created before the error
            return batch_files, total_processed, experiment.experiment_id
        else:
            raise e


def predictions_to_smallpond(experiments, output_path, chunk_size=50000, return_pandas=False, force=False, skip_on_fail=False):
    """
    Pull prediction files from datalake -> smallpond
    """
    import smallpond
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    sp = smallpond.init()
    
    # Determine number of workers
    env_workers = os.environ.get('OE_EVAL_WORKERS')
    num_workers = int(env_workers) if env_workers else min(cpu_count(), len(experiments), 32)
    
    # Get the enforced schema
    schema = get_schema()
    
    # Create temporary directory for batch processing
    temp_dir = Path(tempfile.mkdtemp(prefix="predictions_streaming_mp_"))
    
    try:
        start_time = time.time()
        
        # Prepare arguments for worker processes
        worker_args = [
            (experiment, temp_dir, chunk_size, force, skip_on_fail, i % num_workers)
            for i, experiment in enumerate(experiments)
        ]
        
        # Process experiments in parallel
        all_batch_files = []
        total_processed = 0
        
        with Pool(processes=num_workers) as pool:
            # Use imap for better progress tracking
            results = pool.imap(_process_experiment_worker, worker_args)
            
            for i, (batch_files, processed_count, experiment_id) in enumerate(tqdm(results, total=len(experiments), desc="Processing experiments")):
                all_batch_files.extend(batch_files)
                total_processed += processed_count
                
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"  Completed {i + 1}/{len(experiments)} experiments ({rate:.1f} exp/sec), {total_processed:,} rows processed")
        
        elapsed_time = time.time() - start_time

        print(f"Parallel processing completed in {elapsed_time:.1f} seconds")
        print(f"Created {len(all_batch_files)} batch files with {total_processed:,} total rows")
        
        # Concatenate all batch files efficiently
        print("Combining batch files...")
        final_output = Path(output_path)
        
        # Ensure output directory exists
        final_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Use PyArrow streaming with schema unification for maximum efficiency
        print("Combining batch files using streaming concatenation...")
        
        # Read all tables and unify schema (handle column order differences)
        print(f"Reading {len(all_batch_files)} batch files...")
        all_tables = []
        for batch_file in all_batch_files:
            table = pq.read_table(batch_file)
            all_tables.append(table)
        
        # Concatenate tables
        print("Concatenating tables...")
        unified_table = pa.concat_tables(all_tables)
        
        # Write the unified table
        pq.write_table(unified_table, final_output)
        total_rows = unified_table.num_rows
        
        final_time = time.time() - start_time

        print(f"Total processing time: {final_time / 60:.1f} minutes")
        print(f"Processing rate: {total_rows / final_time:.0f} rows/sec")
        print(f"Streamed {total_rows:,} rows from {len(all_batch_files)} batch files")
        print(f"Final dataset saved to: {final_output}")
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary files")
