#!/bin/bash

# All checkpoints
CHECKPOINTS=(
    # "lucas/olmo2-7b_10b-anneal_s2pdf_base30b_noDDL_plus-dclm-07be83b2"
    #"lucas/olmo2-7b_10b-anneal-all_dclm_baseline-b692c64f"
    # "jakep/olmo2-7b_10b-anneal_s2pdf_fewer_tables_plus-dclm-2ad5861c"
    # "lucas/olmo2-7b_10b-anneal_s2pdf_base30b_noRefs_plus-dclm-6693330d"
    # "jakep/olmo2-7b_10b-anneal_s2pdf_base30b_plus-dclm-c705058f"
    # "jakep/olmo2-7b_10b-anneal_s2pdf_base_plus-dclm-718a6635"
    # "jakep/olmo2-7b_10b-anneal_s2pdf_base30b_md2html_plus-dclm-7e1c8e46"
    # "jakep/olmo2-7b_10b-anneal_s2pdf_base30b_plus-dclm-90cfe702"
    # "jakep/olmo2-7b_10b-anneal_s2pdf_base30b-norefs-and-mdtables_plus-dclm-5f19447f"
    # "jakep/olmo2-7b_10b-anneal_s2pdf_base30b-norefs-and-mdtables-and-noddl_plus-dclm-b27d6561"
   # "jakep/olmo2-7b_10b-anneal_s2pdf_base30b-norefs-and-mdtables_v2_plus-dclm-2f90a9fc"
#    "jakep/OLMo3-1b-5xC-dolma2-s2pdfs-v2-2275d6ea"
#    "jakep/OLMo3-1b-5xC-dolma2-s2pdfs-orig-993eddad"
  # "jakep/olmo2-7b_10b-anneal_s2pdf_base30b-norefs-and-mdtables_v2_finalcheck_plus-dclm-282b0985"
  "jakep/olmo2-7b_10b-anneal_35_65_split_dclm_old_wiki-554d69ad"
  "jakep/olmo2-7b_10b-anneal_35_65_split_dclm_new_wiki-ebf44a53"
)

# Step numbers for each checkpoint (you may need to adjust these)
STEPS=(
    # "step61007"
    # "step61007"
    "step4769"
    "step4769"
)

echo "Starting checkpoint processing..."

# echo "=== COPYING CHECKPOINTS ==="
# for i in "${!CHECKPOINTS[@]}"; do
#     checkpoint="${CHECKPOINTS[$i]}"
#     step="${STEPS[$i]}"
#     echo "Copying checkpoint: $checkpoint at $step"
#     python -m cookbook.remote gs://ai2-llm/checkpoints/$checkpoint/$step/ weka://oe-training-default/ai2-llm/checkpoints/$checkpoint/$step/
# done

# echo "=== CONVERTING CHECKPOINTS ==="
# for i in "${!CHECKPOINTS[@]}"; do
#     checkpoint="${CHECKPOINTS[$i]}"
#     step="${STEPS[$i]}"
#     echo "Converting checkpoint: $checkpoint at $step"
#     olmo-cookbook-eval convert \
#         /oe-training-default/ai2-llm/checkpoints/$checkpoint/$step/ \
#         -t olmo-core-v2 \
#         --use-beaker \
#         --huggingface-tokenizer allenai/dolma2-tokenizer
# done

echo "=== EVALUATING CHECKPOINTS ==="
# Do all the evals
for i in "${!CHECKPOINTS[@]}"; do
    checkpoint="${CHECKPOINTS[$i]}"
    step="${STEPS[$i]}"
    echo "Evaluating checkpoint: $checkpoint at $step"
    olmo-cookbook-eval evaluate "/oe-training-default/ai2-llm/checkpoints/$checkpoint/$step-hf" \
        --tasks "*olmo3:dev:1b:vllm" \
        --priority high \
        --cluster 80g \
        --num-gpus 1 \
        --model-backend vllm \
        --model-args dtype=bfloat16 \
        --partition-size 8 \
        --dashboard olmo-3-evals \
        --workspace ai2/olmo-3-evals
done

# echo "All checkpoint processing completed!"