#!/usr/bin/env bash

configs=(
    'src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/olmo3-nanonanneals-qa-mix/olmo3-nanonanneal-90web-10synthqa.yaml'
    'src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/olmo3-nanonanneals-qa-mix/olmo3-nanonanneal-10web-90synthqa.yaml'
    'src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/olmo3-nanonanneals-qa-mix/olmo3-nanonanneal-20web-80synthqa.yaml'
    'src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/olmo3-nanonanneals-qa-mix/olmo3-nanonanneal-30web-70synthqa.yaml'
    'src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/olmo3-nanonanneals-qa-mix/olmo3-nanonanneal-40web-60synthqa.yaml'
    'src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/olmo3-nanonanneals-qa-mix/olmo3-nanonanneal-50web-50synthqa.yaml'
    'src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/olmo3-nanonanneals-qa-mix/olmo3-nanonanneal-60web-40synthqa.yaml'
    'src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/olmo3-nanonanneals-qa-mix/olmo3-nanonanneal-70web-30synthqa.yaml'
    'src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/olmo3-nanonanneals-qa-mix/olmo3-nanonanneal-80web-20synthqa.yaml'
)

# Launching experiments
for config in "${configs[@]}"; do
    uv run --extra all olmo-cookbook launch -c $config -y
done
