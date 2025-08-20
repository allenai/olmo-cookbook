from collections import defaultdict
import os
import tempfile
import shutil
from pathlib import Path

from tqdm import tqdm


def predictions_to_parquet(predictions):  # List[Predictions]
    """
    Convert a list of predictions to a dataframe, by collapsing the nested dicts to a table. This makes "slicing" data into NumPy arrays easy.
    """
    import pandas as pd  # lazy load

    rows = []
    # task_predictions: Predictions
    for task_predictions in tqdm(predictions, desc="Converting to parquet"):
        metrics = task_predictions.metrics  # MetricsAll

        # prediction: Prediction
        for prediction in task_predictions.predictions:
            # Convert list of dicts to dict of lists for model outputs
            model_output_dict = defaultdict(list)
            for output in prediction.model_output:
                for key, value in output.items():
                    model_output_dict[key].append(value)

            row = {
                "alias": metrics.alias,
                "model_name": metrics.model_name,
                "model_path": metrics.model_path,
                **metrics.to_dict(),
                **prediction.to_dict(),
                **model_output_dict,
            }

            # Expand metrics dict but don't override existing columns
            for metric_key, metric_value in row["metrics"].items():
                if metric_key not in row:
                    row[metric_key] = metric_value

            # For some generation benchmarks, correct_choice is a str, but this will cause a type error
            # when indexing this column
            if "correct_choice" not in row or isinstance(row["correct_choice"], str):
                row["correct_choice"] = 0

            # Sometimes exact_match is bool when it should be float
            if "exact_match" in row and isinstance(row["exact_match"], bool):
                row["exact_match"] = float(row["exact_match"])

            # Standardize cols with multiple dtypes
            if not isinstance(row["native_id"], str):
                row["native_id"] = str(row["native_id"])

            if not isinstance(row["label"], str):
                row["label"] = str(row["label"])

            # Fix legacy names
            #   bits_per_byte_corr -> logits_per_byte_corr
            #   bits_per_byte -> logits_per_byte_corr
            if "bits_per_byte_corr" in row and row["bits_per_byte_corr"] is not None:
                row["logits_per_byte_corr"] = row["bits_per_byte_corr"]

            if "logits_per_byte_corr" in row and row["logits_per_byte_corr"] is not None:
                row["bits_per_byte_corr"] = row["logits_per_byte_corr"]

            # Get the instance-level primary_metric
            primary_metric = row["task_config"]["primary_metric"]
            correct_choice = row["correct_choice"]
            if primary_metric not in row:
                # If the metric doesn't exist (e.g., exact_match), use acc_raw
                if "acc_raw" in row:
                    primary_score = row["acc_raw"]
                elif "exact_match" in row:
                    primary_score = row["exact_match"]
                else:
                    raise ValueError(row)
                assert not isinstance(primary_score, list), row
                row["primary_score"] = primary_score
            elif isinstance(row[primary_metric], list):
                # If the primary_metric is a list (acc_per_char), get the correct choice
                row["primary_score"] = row[primary_metric][correct_choice]
            else:
                row["primary_score"] = row[primary_metric]

            # To save on memory, delete specific fields
            for field in [
                "compute_config",
                "metrics",
                "model_config",
                "task_config",
                "model_output",
                "model_answer",
            ]:
                if field in row:
                    del row[field]

            rows.append(row)

    df = pd.DataFrame(rows)

    # set multiindex
    df.set_index(["alias", "model_name"], inplace=True)

    return df


def predictions_to_smallpond(predictions, output_path=None, chunk_size=50000, return_pandas=False):
    """
    Convert a list of predictions to a smallpond dataset using streaming approach.
    This is a drop-in replacement for predictions_to_parquet that handles large datasets efficiently.
    
    Args:
        predictions: List[Predictions] - The predictions to convert
        output_path: str - Path where to save the final parquet file (optional)
        chunk_size: int - Number of rows to process in each chunk for memory efficiency
        return_pandas: bool - If True, return pandas DataFrame instead of smallpond DataFrame
        
    Returns:
        smallpond.DataFrame or pandas.DataFrame - The resulting dataframe
    """
    import smallpond
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Initialize smallpond session
    sp = smallpond.init()
    
    # Create temporary directory for batch processing
    temp_dir = Path(tempfile.mkdtemp(prefix="predictions_streaming_"))
    
    try:
        # Determine schema by processing first few rows
        print("Analyzing data structure...")
        sample_rows = []
        sample_count = 0
        
        for task_predictions in predictions:
            if sample_count >= 100:  # Sample first 100 rows for schema
                break
            metrics = task_predictions.metrics
            
            for prediction in task_predictions.predictions:
                if sample_count >= 100:
                    break
                    
                # Convert list of dicts to dict of lists for model outputs
                model_output_dict = defaultdict(list)
                for output in prediction.model_output:
                    for key, value in output.items():
                        model_output_dict[key].append(value)

                row = {
                    "alias": metrics.alias,
                    "model_name": metrics.model_name,
                    "model_path": metrics.model_path,
                    **metrics.to_dict(),
                    **prediction.to_dict(),
                    **model_output_dict,
                }

                # Expand metrics dict but don't override existing columns
                for metric_key, metric_value in row["metrics"].items():
                    if metric_key not in row:
                        row[metric_key] = metric_value

                # Apply data cleaning (same as original function)
                if "correct_choice" not in row or isinstance(row["correct_choice"], str):
                    row["correct_choice"] = 0

                if "exact_match" in row and isinstance(row["exact_match"], bool):
                    row["exact_match"] = float(row["exact_match"])

                if not isinstance(row["native_id"], str):
                    row["native_id"] = str(row["native_id"])

                if not isinstance(row["label"], str):
                    row["label"] = str(row["label"])

                # Fix legacy names
                if "bits_per_byte_corr" in row and row["bits_per_byte_corr"] is not None:
                    row["logits_per_byte_corr"] = row["bits_per_byte_corr"]

                if "logits_per_byte_corr" in row and row["logits_per_byte_corr"] is not None:
                    row["bits_per_byte_corr"] = row["logits_per_byte_corr"]

                # Get the instance-level primary_metric
                primary_metric = row["task_config"]["primary_metric"]
                correct_choice = row["correct_choice"]
                if primary_metric not in row:
                    if "acc_raw" in row:
                        primary_score = row["acc_raw"]
                    elif "exact_match" in row:
                        primary_score = row["exact_match"]
                    else:
                        raise ValueError(row)
                    assert not isinstance(primary_score, list), row
                    row["primary_score"] = primary_score
                elif isinstance(row[primary_metric], list):
                    row["primary_score"] = row[primary_metric][correct_choice]
                else:
                    row["primary_score"] = row[primary_metric]

                # Fix data types at the source to ensure consistency (same as main processing)
                # Nullable float columns
                nullable_float_columns = ['predicted_index_uncond', 'acc_uncond']
                for col in nullable_float_columns:
                    if col in row:
                        if row[col] is not None:
                            try:
                                row[col] = float(row[col])
                            except (ValueError, TypeError):
                                row[col] = None
                
                # Integer columns that should be consistent
                int_columns = ['task_idx', 'doc_id', 'num_instances', 'correct_choice', 
                              'predicted_index_raw', 'predicted_index_per_token', 
                              'predicted_index_per_char', 'predicted_index_per_byte',
                              'acc_raw', 'acc_per_token', 'acc_per_char', 'acc_per_byte', 'no_answer']
                for col in int_columns:
                    if col in row and row[col] is not None:
                        try:
                            row[col] = int(float(row[col]))  # Convert via float to handle string numbers
                        except (ValueError, TypeError):
                            row[col] = 0
                
                # Float columns that should be consistent
                float_columns = ['processing_time', 'sum_logits_corr', 'logits_per_token_corr',
                               'logits_per_char_corr', 'bits_per_byte_corr', 'logits_per_byte_corr', 'primary_score']
                for col in float_columns:
                    if col in row and row[col] is not None:
                        try:
                            row[col] = float(row[col])
                        except (ValueError, TypeError):
                            row[col] = 0.0

                # Remove memory-heavy fields
                for field in [
                    "compute_config",
                    "metrics", 
                    "model_config",
                    "task_config",
                    "model_output",
                    "model_answer",
                ]:
                    if field in row:
                        del row[field]

                sample_rows.append(row)
                sample_count += 1
        
        # Create schema from sample
        sample_df = pd.DataFrame(sample_rows)
        schema = pa.Schema.from_pandas(sample_df)
        print(f"Detected schema with {len(schema)} columns")
        
        # Process data in streaming chunks
        batch_files = []
        current_chunk = []
        total_processed = 0
        
        print("Processing predictions in streaming chunks...")
        for task_predictions in tqdm(predictions, desc="Converting to smallpond"):
            metrics = task_predictions.metrics
            
            for prediction in task_predictions.predictions:
                # Convert list of dicts to dict of lists for model outputs
                model_output_dict = defaultdict(list)
                for output in prediction.model_output:
                    for key, value in output.items():
                        model_output_dict[key].append(value)

                row = {
                    "alias": metrics.alias,
                    "model_name": metrics.model_name,
                    "model_path": metrics.model_path,
                    **metrics.to_dict(),
                    **prediction.to_dict(),
                    **model_output_dict,
                }

                # Expand metrics dict but don't override existing columns
                for metric_key, metric_value in row["metrics"].items():
                    if metric_key not in row:
                        row[metric_key] = metric_value

                # Apply all the same data cleaning as original function
                if "correct_choice" not in row or isinstance(row["correct_choice"], str):
                    row["correct_choice"] = 0

                if "exact_match" in row and isinstance(row["exact_match"], bool):
                    row["exact_match"] = float(row["exact_match"])

                if not isinstance(row["native_id"], str):
                    row["native_id"] = str(row["native_id"])

                if not isinstance(row["label"], str):
                    row["label"] = str(row["label"])

                # Fix legacy names
                if "bits_per_byte_corr" in row and row["bits_per_byte_corr"] is not None:
                    row["logits_per_byte_corr"] = row["bits_per_byte_corr"]

                if "logits_per_byte_corr" in row and row["logits_per_byte_corr"] is not None:
                    row["bits_per_byte_corr"] = row["logits_per_byte_corr"]

                # Get the instance-level primary_metric
                primary_metric = row["task_config"]["primary_metric"]
                correct_choice = row["correct_choice"]
                if primary_metric not in row:
                    if "acc_raw" in row:
                        primary_score = row["acc_raw"]
                    elif "exact_match" in row:
                        primary_score = row["exact_match"]
                    else:
                        raise ValueError(row)
                    assert not isinstance(primary_score, list), row
                    row["primary_score"] = primary_score
                elif isinstance(row[primary_metric], list):
                    row["primary_score"] = row[primary_metric][correct_choice]
                else:
                    row["primary_score"] = row[primary_metric]

                # Fix data types at the source to ensure consistency
                # Nullable float columns
                nullable_float_columns = ['predicted_index_uncond', 'acc_uncond']
                for col in nullable_float_columns:
                    if col in row:
                        if row[col] is not None:
                            try:
                                row[col] = float(row[col])
                            except (ValueError, TypeError):
                                row[col] = None
                
                # Integer columns that should be consistent
                int_columns = ['task_idx', 'doc_id', 'num_instances', 'correct_choice', 
                              'predicted_index_raw', 'predicted_index_per_token', 
                              'predicted_index_per_char', 'predicted_index_per_byte',
                              'acc_raw', 'acc_per_token', 'acc_per_char', 'acc_per_byte', 'no_answer']
                for col in int_columns:
                    if col in row and row[col] is not None:
                        try:
                            row[col] = int(float(row[col]))  # Convert via float to handle string numbers
                        except (ValueError, TypeError):
                            row[col] = 0
                
                # Float columns that should be consistent
                float_columns = ['processing_time', 'sum_logits_corr', 'logits_per_token_corr',
                               'logits_per_char_corr', 'bits_per_byte_corr', 'logits_per_byte_corr', 'primary_score']
                for col in float_columns:
                    if col in row and row[col] is not None:
                        try:
                            row[col] = float(row[col])
                        except (ValueError, TypeError):
                            row[col] = 0.0

                # Remove memory-heavy fields
                for field in [
                    "compute_config",
                    "metrics",
                    "model_config", 
                    "task_config",
                    "model_output",
                    "model_answer",
                ]:
                    if field in row:
                        del row[field]

                current_chunk.append(row)
                total_processed += 1
                
                # Write chunk when it reaches target size
                if len(current_chunk) >= chunk_size:
                    batch_file = temp_dir / f"batch_{len(batch_files):06d}.parquet"
                    
                    # Convert to PyArrow table via pandas (for schema consistency)
                    chunk_df = pd.DataFrame(current_chunk)
                    table = pa.Table.from_pandas(chunk_df)
                    pq.write_table(table, batch_file)
                    
                    batch_files.append(batch_file)
                    current_chunk = []
                    
                    if len(batch_files) % 10 == 0:
                        print(f"  Processed {total_processed:,} rows in {len(batch_files)} chunks")
        
        # Write final chunk if any remaining data
        if current_chunk:
            batch_file = temp_dir / f"batch_{len(batch_files):06d}.parquet"
            
            # Convert to PyArrow table via pandas (for schema consistency)
            chunk_df = pd.DataFrame(current_chunk)
            table = pa.Table.from_pandas(chunk_df)
            pq.write_table(table, batch_file)
            
            batch_files.append(batch_file)
        
        print(f"Created {len(batch_files)} batch files with {total_processed:,} total rows")
        
        # Concatenate all batch files efficiently
        print("Combining batch files...")
        if output_path:
            final_output = Path(output_path)
        else:
            final_output = Path("predictions_combined.parquet")
        
        # Ensure output directory exists
        final_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Use PyArrow streaming with schema unification for maximum efficiency
        print("Combining batch files using streaming concatenation...")
        
        # Read all tables and unify schema (handle column order differences)
        all_tables = []
        for i, batch_file in enumerate(batch_files):
            table = pq.read_table(batch_file)
            all_tables.append(table)
            
            if i == 0:
                print(f"First batch: {table.num_rows} rows, {len(table.columns)} columns")
            elif i < 3:
                print(f"Batch {i}: {table.num_rows} rows, {len(table.columns)} columns")
        
        # Unify schemas and concatenate using PyArrow
        print("Unifying schemas and concatenating...")
        try:
            # PyArrow can handle schema differences with concat_tables
            unified_table = pa.concat_tables(all_tables, promote_options='default')
            
            # Write the unified table directly
            pq.write_table(unified_table, final_output)
            total_rows = unified_table.num_rows
            
        except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
            # Fallback to pandas for complex schema differences
            print(f"PyArrow concat failed ({str(e)[:100]}...), falling back to pandas concatenation")
            dataframes = []
            for table in all_tables:
                df = table.to_pandas()
                dataframes.append(df)
            
            # Use pandas to handle schema differences gracefully
            combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
            
            # Set multiindex before converting to PyArrow
            combined_df = combined_df.set_index(["alias", "model_name"])
            
            # Convert back to PyArrow and write
            unified_table = pa.Table.from_pandas(combined_df)
            pq.write_table(unified_table, final_output)
            total_rows = len(combined_df)
        
        print(f"Streamed {total_rows:,} rows from {len(batch_files)} batch files")
        
        print(f"Final dataset saved to: {final_output}")
        
        if return_pandas:
            # Read the parquet file back as pandas DataFrame (multiindex already set if using pandas fallback)
            pandas_df = pd.read_parquet(final_output)
            # Check if multiindex is already set
            if not isinstance(pandas_df.index, pd.MultiIndex):
                pandas_df = pandas_df.set_index(["alias", "model_name"])
            return pandas_df
        else:
            # Load into smallpond and set multiindex
            combined_df = sp.read_parquet(str(final_output))
            combined_df = combined_df.set_index(["alias", "model_name"])
            return combined_df
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary files")
