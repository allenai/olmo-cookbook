#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scipy",
#     "numpy",
# ]
# ///

import json
import sys
from scipy.stats import spearmanr, pearsonr
import numpy as np




def main():

    try:
        data: dict[str, dict[str, float | None]] = json.loads(sys.stdin.read())    # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        print(
            "You must pipe a JSON object to this script, e.g. " +
            "`olmo-cookbook-eval results --dashboard olmo3-midtraining-web --task olmo3:dev:midtrain:v0 " +
            "| ./scripts/rank_correlation.py`",
            file=sys.stderr,
        )
        return

    # Identify metric pairs
    metric_pairs: dict[str, tuple[str, str]] = {}
    for metric in data:
        if metric.endswith("::xlarge"):
            base_metric = metric[:-8]  # Remove "::xlarge"
            mc_metric = base_metric + ":mc::gen2mc"
            if mc_metric in data:
                metric_pairs[base_metric] = (metric, mc_metric)
        elif metric.endswith(":mc::gen2mc"):
            base_metric = metric[:-11]  # Remove ":mc::gen2mc"
            xlarge_metric = base_metric + "::xlarge"
            if xlarge_metric in data and base_metric not in metric_pairs:
                metric_pairs[base_metric] = (xlarge_metric, metric)

    if not metric_pairs:
        print("No paired metrics found with both ::xlarge and :mc::gen2mc suffixes")
        return

    # Calculate correlations for each metric pair
    results = []
    for base_metric, (xlarge_metric, mc_metric) in metric_pairs.items():
        xlarge_data = data[xlarge_metric]
        mc_data = data[mc_metric]

        # Find common models with non-null values
        common_models: list[str] = []
        for model in sorted(set(xlarge_data.keys()) & set(mc_data.keys())):
            if xlarge_data[model] is not None and mc_data[model] is not None:
                common_models.append(model)

        if len(common_models) < 2:
            print(f"Skipping {base_metric}: insufficient common models with non-null values ({len(common_models)})")
            continue

        # Extract values for common models
        xlarge_values = [xlarge_data[model] for model in common_models]
        mc_values = [mc_data[model] for model in common_models]

        # Calculate correlations
        spearman_corr, spearman_p = spearmanr(xlarge_values, mc_values)
        pearson_corr, pearson_p = pearsonr(xlarge_values, mc_values)

        results.append({
            "metric": base_metric,
            "n_models": len(common_models),
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "models": common_models
        })

    # Print results
    print(f"\nFound {len(results)} metric pairs with correlations:\n")

    for result in sorted(results, key=lambda x: abs(x["spearman_correlation"]), reverse=True):
        print(f"Metric: {result['metric']}")
        print(f"  Models compared: {result['n_models']}")
        print(f"  Spearman correlation: {result['spearman_correlation']:.4f} (p={result['spearman_p_value']:.4f})")
        print(f"  Pearson correlation: {result['pearson_correlation']:.4f} (p={result['pearson_p_value']:.4f})")
        print()

if __name__ == "__main__":
    main()
