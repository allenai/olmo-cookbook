import warnings

import numpy as np
import pandas as pd


def get_slice(df, model=None, task=None):
    """Index to return a df of some (model, task)"""
    models = [model] if isinstance(model, str) else model
    tasks = [task] if isinstance(task, str) else task

    # Dynamically create a slicing tuple matching the index levels
    level_slices = {
        "model_name": models if models else slice(None),
        "alias": tasks if tasks else slice(None),
    }
    slicing_tuple = tuple(level_slices.get(level, slice(None)) for level in df.index.names)

    is_multiindex = isinstance(df.index, pd.MultiIndex)

    if is_multiindex:
        try:
            # df = df.loc[slicing_tuple]
            idx = pd.MultiIndex.from_product(slicing_tuple, names=["alias", "model_name"])
            df = df.loc[idx]
        except KeyError:
            return df.iloc[0:0]  # Return an empty DataFrame if no match
    else:
        # Slow index
        df = df[
            (
                df["model_name"].isin(level_slices["model_name"])
                if isinstance(level_slices["model_name"], list)
                else True
            )
            & (df["alias"].isin(level_slices["alias"]) if isinstance(level_slices["alias"], list) else True)
        ]

    df = df.reset_index()

    return df


def get_nd_array(df, col, metric, model=None, task=None, sorted=False):
    """Get an nd array of (COL, instances), sorted by overall performance"""
    col = [col] if not isinstance(col, list) else col

    slices = get_slice(df, model, task)

    if len(slices) == 0:
        return [], np.array([])

    is_multiindex = isinstance(df.index, pd.MultiIndex)

    if is_multiindex:
        # For native_ids which count up from 0, there are the same IDs across tasks. Append the task name.
        slices["native_id"] = slices["native_id"] + "_" + slices["alias"].astype(str)

        duplicates_count = slices.duplicated(subset=["native_id"] + col).sum()
        if duplicates_count > 0 and task is not None:
            warnings.simplefilter("once", UserWarning)
            warnings.warn(
                f"Warning: {duplicates_count}/{len(slices)} duplicate native_id-key pairs found for task='{task}' model='{model}'. Removing duplicates...",
                category=UserWarning,
                stacklevel=2,
            )
            slices = slices.sort_values(by=col, na_position="last").drop_duplicates(
                subset=["native_id"] + col, keep="first"
            )

        # Pivot the data to get mixes as columns and question_ids as rows
        pivoted = slices.pivot(index="native_id", columns=col, values=metric)

        columns = pivoted.columns
        scores = pivoted.to_numpy()
    else:
        pivoted = None
        if len(col) == 1:
            columns = slices[col[0]].to_numpy()
            scores = slices[metric].to_numpy()
        else:
            pivoted = slices.pivot(index=col[0], columns=col[1:], values=metric)
            columns = pivoted.columns
            scores = pivoted.to_numpy()

    if is_multiindex and pivoted is not None:
        # If there are multiple cols, reshape the output nd array
        if len(col) > 1:
            pivoted = pivoted.sort_index(axis=1)
            expanded_columns = pivoted.columns.to_frame(index=False)
            pivoted.columns = pd.MultiIndex.from_tuples(
                [tuple(col) for col in expanded_columns.to_numpy()], names=expanded_columns.columns.tolist()
            )
            scores = pivoted.to_numpy()
            unique_counts = [len(expanded_columns[level].unique()) for level in expanded_columns.columns]
            scores = scores.reshape((pivoted.shape[0], *unique_counts))

        # # Add a new axis for dim=1 if necessary
        # scores = np.expand_dims(scores, axis=1)

    # Move instances dim to final dim
    scores = np.moveaxis(scores, 0, -1)

    if sorted:
        if len(col) == 1 and not is_multiindex:
            sorted_indices = np.argsort(scores)
            columns = columns[sorted_indices]
            scores = scores[sorted_indices]
        else:
            # Sort by overall performance
            mix_sums = scores.sum(axis=1)
            sorted_indices = mix_sums.argsort()[::-1]
            columns = columns[sorted_indices].tolist()
            scores = scores[sorted_indices]

    if not isinstance(columns, list):
        columns = columns.tolist()

    return columns, scores
