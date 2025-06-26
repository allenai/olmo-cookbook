from collections import defaultdict

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
