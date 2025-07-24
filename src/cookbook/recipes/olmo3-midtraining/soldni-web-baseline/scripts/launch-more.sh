configs=(
    "src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/more-jeopardy-plz/olmo3-microanneal-4T-50v20-50flan.yaml"
    "src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/more-jeopardy-plz/olmo3-microanneal-4T-50v20-50synth-qa-wiki-rewrite.yaml"
    "src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/more-jeopardy-plz/olmo3-microanneal-4T-round1-flan-synthqa.yaml"
    "src/cookbook/recipes/olmo3-midtraining/soldni-web-baseline/configs/more-jeopardy-plz/olmo3-microanneal-4T-round1-flan.yaml"
)

for config in "${configs[@]}"; do
    uv run --extra=all --python 3.12 olmo-cookbook launch -c "${config}" -y
done
