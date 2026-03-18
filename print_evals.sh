DASHBOARD="kevinf-flex"
uv run olmo-cookbook-eval results \
    --dashboard "${DASHBOARD}" \
    --tasks mt_mbpp_v2fix \
    --tasks code-no-bcb \
    -T --force
