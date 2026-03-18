DASHBOARD="kevinf-flex"
uv run olmo-cookbook-eval results \
    --dashboard "${DASHBOARD}" \
    --tasks mt_mbpp_v2fix \
    --tasks code-no-bcb \
    -T \
    --lower-is-better \
    -x "olmo3_1b_5xc_50web_alldressed_v2_50spring2code_stack_edu_redux"
    # -x "train-olmo3-1b-dolma50-swallow-python50-10B-lr5e-5-ctd_step2385-hf" \
    # --models "new-kevinf-olmo3"  \
    # --models "train-olmo3-1b-dolma50-stackedu"  \
