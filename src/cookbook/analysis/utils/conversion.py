from collections import defaultdict
import json
import logging
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import List

import numpy as np
from tqdm import tqdm

from cookbook.eval.datalake import Instance, Instances, InstancesAll, MetricsAll, Prediction, Predictions, PredictionsAll

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def get_predictions_schema():
    """Instance-level prediction table schema"""
    return {
        'task_alias': str,
        'model_name': str,
        'model_revision': str,
        'instance_id': str,
        'primary_score': float,
        'bits_per_byte_corr': float,
        'pass_at_1': float,
        'exact_match': float,
        'exact_match_flex': float,
        'f1': float,
        'recall': float,
        'label': str,
        'correct_choice': int,
        'model_answer': str,
        'internal_model_name': str,
        'doc_id': int,
        'native_id': str,

        # 'pass_at_2': float,
        # 'pass_at_4': float,
        # 'pass_at_10': float,
        # 'pass_at_16': float,
        # 'pass_at_32': float,
        # 'maj_at_1': float,
        # 'maj_at_2': float,
        # 'maj_at_4': float,
        # 'maj_at_16': float,
        # 'maj_at_32': float,

        # 'logits_per_char': List[float],
        # 'bits_per_byte': List[float],
        # 'logits_per_char_corr': float,
        # 'logits_per_byte_corr': float,

        # num_instances
        # processing_time
        # continuation
        # tested_completion
        # exec_result

        # 'num_tokens': int,
        # sum_logits
        # num_tokens_all
        # predicted_index_raw
        # predicted_index_per_token
        # predicted_index_per_char
        # predicted_index_per_byte
        # predicted_index_uncond
        # acc_raw
        # acc_per_token
        # acc_per_char
        # acc_per_byte
        # acc_uncond
        # no_answer
        # rank
        # greedy_acc
    }


def get_questions_schema():
    """Questions table schema"""
    return {
        'task_alias': str,
        'instance_id': str,
        'doc': str,
        'label': str,
        'task_name': str,
        'doc_id': int,
        'native_id': str
    }


def get_instance_id(pred):
    # @david -- this handling is blursed. See python -c "print(type(str(None)))"
    if pred["native_id"] is not None and pred["native_id"] != "None":
        return str(pred["native_id"])
    elif pred["doc_id"] is not None and pred["doc_id"] != "None":
        return str(pred["doc_id"])


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
    import pandas as pd

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


def _process_prediction_row(prediction: Prediction, metrics: MetricsAll):
    """
    Process a single prediction row with common data cleaning logic.
    Returns the processed row ready for schema enforcement.
    """
    # Convert list of dicts to dict of lists for model outputs
    model_output_dict = defaultdict(list)
    for output in prediction.model_output:
        for key, value in output.items():
            model_output_dict[key].append(value)

    # Build the full row first
    full_row = {
        "task_alias": metrics.alias,
        "task_name": metrics.task_name,
        "model_name": metrics.model_config.get("model_path", None),
        "model_revision": metrics.model_config.get("revision", None),
        "internal_model_name": metrics.model_config.get("model", None),
        **metrics.to_dict(),
        **model_output_dict,
    }

    # Add prediction dict
    full_row.update(prediction.to_dict())

    # Expand metrics dict but don't override existing columns
    for metric_key, metric_value in full_row["metrics"].items():
        if metric_key not in full_row:
            full_row[metric_key] = metric_value

    # Apply data cleaning for schema compatibility
    if "correct_choice" not in full_row or isinstance(full_row["correct_choice"], str):
        full_row["correct_choice"] = np.nan

    if "exact_match" in full_row and isinstance(full_row["exact_match"], bool):
        full_row["exact_match"] = float(full_row["exact_match"])

    if "exact_match_flex" in full_row and isinstance(full_row["exact_match_flex"], bool):
        full_row["exact_match_flex"] = float(full_row["exact_match_flex"])
        
    if not isinstance(full_row["native_id"], str):
        full_row["native_id"] = str(full_row["native_id"])

    if not isinstance(full_row["label"], str):
        full_row["label"] = str(full_row["label"])

    # Set default model_revision to "main"
    if not full_row.get("model_revision"):
        full_row["model_revision"] = "main"

    # Fix legacy names
    if "bits_per_byte_corr" in full_row and full_row["bits_per_byte_corr"] is not None:
        full_row["logits_per_byte_corr"] = full_row["bits_per_byte_corr"]

    if "logits_per_byte_corr" in full_row and full_row["logits_per_byte_corr"] is not None:
        full_row["bits_per_byte_corr"] = full_row["logits_per_byte_corr"]
    
    # Get instance ID
    full_row["instance_id"] = get_instance_id(full_row)

    # Get the instance-level primary_metric
    primary_metric = full_row["task_config"]["primary_metric"]
    correct_choice = full_row["correct_choice"]
    if primary_metric not in full_row:
        # Unfortunately, not all oe-eval tasks specify a metric, so default to these
        if "acc_per_char" in full_row:
            primary_score = full_row["acc_per_char"]
        elif "exact_match_flex" in full_row:
            primary_score = full_row["exact_match_flex"]
        elif "exact_match" in full_row:
            primary_score = full_row["exact_match"]
        elif "pass_at_1" in full_row:
            primary_score = full_row["pass_at_1"]
        else:
            raise ValueError("Could not get primary metric for entry: " + _truncate_for_display(full_row) + f" with keys: {full_row.keys()} and with metrics: {full_row['metrics']}")
        assert not isinstance(primary_score, list), _truncate_for_display(full_row)
        full_row["primary_score"] = primary_score
    elif isinstance(full_row[primary_metric], list):
        choice_index = correct_choice if not np.isnan(correct_choice) else 0
        full_row["primary_score"] = full_row[primary_metric][choice_index]
    else:
        full_row["primary_score"] = full_row[primary_metric]

    return full_row


def _process_question_row(instance: Instance, metrics: MetricsAll):
    full_row = {
        "task_alias": metrics.alias,
        "task_name": metrics.task_name,
        **metrics.to_dict(),
    }

    full_row["native_id"] = str(instance.native_id)
    full_row["doc_id"] = str(instance.doc_id)
    full_row["label"] = str(instance.label)
    full_row["instance_id"] = get_instance_id(full_row)

    # @davidh -- I would like to store this primitive as a dict
    full_row["doc"] = str(instance.doc)

    return full_row


def _get_type_mapping(schema_func):
    """
    Get a mapping of field names to their required PyArrow types based on the enforced schema.
    """
    import pyarrow as pa
    
    schema = schema_func()
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


def _create_consistent_prediction_table(chunk_df, worker_id=None):
    return _create_consistent_table_generic(chunk_df, _get_type_mapping(get_predictions_schema), worker_id)


def _create_consistent_questions_table(chunk_df, worker_id=None):
    return _create_consistent_table_generic(chunk_df, _get_type_mapping(get_questions_schema), worker_id)


def _create_consistent_table_generic(chunk_df, type_mapping, worker_id=None):
    """
    Create a PyArrow table with consistent types for the given type mapping.
    """
    import pyarrow as pa
    
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


def _generic_experiment_worker(args):
    """
    Worker function for processing a single experiment in parallel with enforced schema.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    import time

    experiment, data_type, task_aliases, temp_dir, chunk_size, force, skip_on_fail, worker_id = args
    
    # Select schema and processing functions based on data type
    if data_type == "predictions":
        schema = get_predictions_schema()
        data_fetcher = PredictionsAll.run
        process_row_func = _process_prediction_row
        table_creator = _create_consistent_prediction_table
    elif data_type == "instances":
        schema = get_questions_schema()
        data_fetcher = InstancesAll.run
        process_row_func = _process_question_row
        table_creator = _create_consistent_questions_table
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    batch_files = []
    current_chunk = []
    total_processed = 0
    
    # Create unique batch counter using timestamp and experiment ID to avoid filename collisions
    batch_counter = 0
    unique_prefix = f"{int(time.time() * 1000000)}_{experiment.experiment_id[:8]}"
    
    try:
        all_data = data_fetcher(experiment_id=experiment.experiment_id, force=force, skip_on_fail=skip_on_fail)

        for data_collection in all_data:
            metrics: MetricsAll = data_collection.metrics
            task_alias = metrics.alias

            if task_aliases is not None and task_alias not in task_aliases:
                # Only include requested tasks
                continue
            
            if data_type == "predictions":
                data_items: Predictions = data_collection.predictions
            elif data_type == "instances":
                data_items: Instances = data_collection.instances
            else:
                raise ValueError(data_type)

            for data_item in data_items:
                try:
                    # Process data item
                    full_row = process_row_func(data_item, metrics)
                    
                    # Enforce the schema
                    row = _enforce_schema(full_row, schema)
                    current_chunk.append(row)
                    total_processed += 1
                    
                    # Write chunk when it reaches target size
                    if len(current_chunk) >= chunk_size:
                        batch_file = temp_dir / f"worker_{worker_id}_{unique_prefix}_batch_{batch_counter:06d}.parquet"
                        batch_counter += 1
                        
                        # Create table
                        chunk_df = pd.DataFrame(current_chunk)
                        table = table_creator(chunk_df, worker_id=worker_id)
                        pq.write_table(table, batch_file)
                        
                        batch_files.append(batch_file)
                        current_chunk = []
                
                except Exception as e:
                    if skip_on_fail:
                        if "404 client error" in str(e).lower():
                            # Some files simply don't exist in the datalake
                            continue
                        logger.info(f"[Worker {worker_id}] ⚠️  Task {metrics.task_idx} in experiment {experiment.experiment_id} failed (skipping): {e}")
                        continue
                    else:
                        raise e
        
        # Write final chunk if any remaining data
        if current_chunk:
            batch_file = temp_dir / f"worker_{worker_id}_{unique_prefix}_batch_{batch_counter:06d}.parquet"
            
            # Create table
            chunk_df = pd.DataFrame(current_chunk)
            table = table_creator(chunk_df, worker_id=worker_id)
            pq.write_table(table, batch_file)
            
            batch_files.append(batch_file)
        
        return batch_files, total_processed, experiment.experiment_id
        
    except Exception as e:
        if skip_on_fail:
            logger.info(f"[Worker {worker_id}] ❌ Experiment {experiment.experiment_id} failed (Processed {total_processed} rows before failure): {e}")

            # Return any batch files that were successfully created before the error
            return batch_files, total_processed, experiment.experiment_id
        else:
            raise e



def construct_smallpond(experiments, task_aliases, output_path, data_type, chunk_size=50000, return_pandas=False, force=False, skip_on_fail=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
     
    # Determine number of workers
    env_workers = os.environ.get('OE_EVAL_WORKERS')
    num_workers = int(env_workers) if env_workers else min(cpu_count(), len(experiments), 64)
    
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{data_type}_streaming_mp_"))
    
    try:
        start_time = time.time()
        
        # Prepare arguments for workers
        worker_args = [
            (experiment, data_type, task_aliases, temp_dir, chunk_size, force, skip_on_fail, i % num_workers)
            for i, experiment in enumerate(experiments)
        ]
        
        # Process experiments
        all_batch_files = []
        total_processed = 0
        
        with Pool(processes=num_workers) as pool:
            results = pool.imap(_generic_experiment_worker, worker_args)
            
            pbar = tqdm(results, total=len(experiments), desc=f"Processing {data_type}")
            for i, (batch_files, processed_count, experiment_id) in enumerate(pbar):
                all_batch_files.extend(batch_files)
                total_processed += processed_count
                
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    pbar.set_description(f"Processing {data_type} ({rate:.1f} exp/sec, {total_processed:,} rows)")
        
        elapsed_time = time.time() - start_time

        logger.info(f"Parallel processing completed in {elapsed_time:.1f} seconds")
        logger.info(f"Created {len(all_batch_files)} batch files with {total_processed:,} total rows")
        
        # Concatenate all batch files efficiently
        final_output = Path(output_path)
        final_output.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Reading batch files...")
        all_tables = []
        for batch_file in all_batch_files:
            table = pq.read_table(batch_file)
            all_tables.append(table)
        
        # Concatenate tables
        if not all_tables:
            logger.warning("No data was produced from any experiment. Output file will not be created.")
            return
        logger.info("Concatenating tables...")
        unified_table = pa.concat_tables(all_tables)
        pq.write_table(unified_table, final_output)

        logger.info(f"Total processing time: {(time.time() - start_time) / 60:.1f} minutes")
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")
