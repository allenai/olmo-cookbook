group_id=6d2d4c39 # 24
# group_id=8b10a86d # 48

for i in $(seq -f "%04g" 0 23); do
    echo "Syncing checkpoint part-$i..."
    python -m cookbook.remote \
        s3://ai2-llm/checkpoints/ai2-tylerm/olmo2-pdfs-datadelve-5xC-30m-augusta-2048-$group_id-$i/step22100/ \
        weka://oe-training-default/ai2-llm/checkpoints/ai2-tylerm/olmo2-pdfs-datadelve-5xC-30m-augusta-2048-$group_id-$i/step22100 \
        --workspace ai2/dolma2
done

# for i in $(seq -f "%04g" 0 47); do
#   echo "Converting checkpoint $i..."
#   olmo-cookbook-eval convert "/oe-training-default/ai2-llm/checkpoints/ai2-tylerm/olmo2-pdfs-datadelve-5xC-30m-augusta-2048-${group_id}-${i}/step22100" \
#     -t olmo-core-v2 \
#     --use-beaker \
#     --beaker-workspace ai2/dolma2
# done

# for i in "${INCLUDED[@]}"; do
#     CHECKPOINT="/oe-data-default/ai2-llm/checkpoints/ai2-tylerm/olmo2-pdfs-datadelve-5xC-30m-augusta-2048-${group_id}-${i}/step22100-hf"

#     echo "Evaluating checkpoint: $CHECKPOINT"

#     olmo-cookbook-eval evaluate "$CHECKPOINT" \
#         --tasks arc_easy:rc::olmes:full \
#         --tasks arc_challenge:rc::olmes:full \
#         --tasks hellaswag:rc::olmes \
#         --tasks winogrande:rc::olmes:full \
#         --tasks csqa:rc::olmes:full \
#         --tasks piqa:rc::olmes:full \
#         --tasks socialiqa:rc::olmes:full \
#         --tasks mmlu:rc \
#         --tasks basic_skills:rc::olmes \
#         --tasks minerva \
#         --tasks codex_humaneval:3shot:bpb::none \
#         --tasks mbpp:3shot:bpb::none \
#         --tasks mt_mbpp \
#         --tasks medmcqa:rc:bpb::none \
#         --tasks lambada \
#         --tasks sciq:bpb::olmo1 \
#         --tasks squad:rc:bpb::gen2mc \
#         --tasks naturalqs:rc:bpb::gen2mc  \
#         --tasks jeopardy:rc:bpb::gen2mc \
#         --tasks drop:rc:bpb::gen2mc \
#         --tasks coqa:rc:bpb::gen2mc \
#         --compute-gold-bpb \
#         --priority urgent \
#         --cluster 80g \
#         --num-gpus 1 \
#         --model-backend vllm \
#         --model-args dtype=bfloat16 \
#         --partition-size 8 \
#         --dashboard regmixer \
#         --workspace ai2/dolma2

#     olmo-cookbook-eval evaluate "$CHECKPOINT" \
#         --tasks ultrachat_masked_ppl \
#         --tasks wildchat_masked_ppl \
#         --compute-gold-bpb \
#         --priority urgent \
#         --cluster 80g \
#         --num-gpus 1 \
#         --model-backend hf \
#         --model-args dtype=bfloat16 \
#         --partition-size 8 \
#         --dashboard regmixer \
#         --workspace ai2/dolma2
# done

