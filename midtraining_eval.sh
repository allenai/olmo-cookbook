#!/bin/bash

count_jobs() {
    jobs -r | wc -l
}

: 'for i in $(seq -w 0 15); do
  while [ $(count_jobs) -ge 12 ]; do
    sleep 5
  done

  echo "Starting experiment for index $i" 
  {
    exp_id="olmo3_7b-12T-5B-round-2-code-conditional-gen-mcqa-swarm-515eaf2d-$i"
    echo $exp_id

    olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step2385-hf" \
    --tasks olmo3:dev:7b:main:v2 \
    --priority high \
    --cluster ai2/jupiter-cirrascale-2 \
    --partition-size 8 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args trust_remote_code=true,max_length=4096 \
    --beaker-image oe-eval-beaker/oe_eval_olmo3_auto\
    --fim-tokens l2c \
    --vllm-use-v1-spec \
    --vllm-memory-utilization 0.7 \
    --dashboard olmo3-midtraining-mixing \
    --workspace ai2/oe-data \
    --budget ai2/oe-base \
    --priority urgent
  } &
done

wait 
echo "All experiments completed."
'


: 'for i in $(seq -w 0 47); do
  while [ $(count_jobs) -ge 12 ]; do
    sleep 5
  done

  echo "Starting experiment for index $i" 
  {
    exp_id="olmo3_7b-12T-5B-round-2-code-math-conditional-gen-mcqa-swarm-a3e06472-$i"
    echo $exp_id

    olmo-cookbook-eval evaluate \
    "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step2385-hf" \
    --tasks olmo3:dev:7b:main:v2 \
    --priority high \
    --cluster ai2/jupiter-cirrascale-2 \
    --partition-size 8 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args trust_remote_code=true,max_length=4096 \
    --beaker-image oe-eval-beaker/oe_eval_olmo3_auto\
    --fim-tokens l2c \
    --vllm-use-v1-spec \
    --vllm-memory-utilization 0.7 \
    --dashboard olmo3-midtraining-mixing \
    --workspace ai2/oe-data \
    --budget ai2/oe-base \
    --priority urgent
  } &
done

wait 
echo "All experiments completed."
'





 experiments=(
  olmo3_7b-12T-5B-round-3-sweep-gen-mc-code-math-30B-a0160e7d
  olmo3_7b-12T-5B-round-3-sweep-gen-mc-code-math-35B-37f49e0b
  olmo3_7b-12T-5B-round-3-sweep-gen-mc-code-math-40B-3b0ea5e3
  olmo3_7b-12T-5B-round-3-sweep-gen-mc-code-math-45B-88bc0109
  olmo3_7b-12T-5B-round-3-sweep-gen-mc-code-math-50B-de5ea711
  olmo3_7b-12T-5B-round-3-sweep-gen-mc-code-math-55B-bb2e09e0
  olmo3_7b-12T-5B-round-3-sweep-gen-mc-code-math-60B-931e11dc
)

for exp_id in "${experiments[@]}"; do
  while [ $(count_jobs) -ge 12 ]; do
    sleep 5
  done

  echo "Starting experiment $exp_id" 
  {

    olmo-cookbook-eval evaluate \
      "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step2385-hf" \
      --tasks olmo3:dev:midtrain:v1 \
      --priority urgent \
      --cluster ai2/jupiter-cirrascale-2 \
      --num-gpus 1 \
      --partition-size 8 \
      --model-backend vllm \
      --no-compute-gold-bpb \
      --model-args chat_template=basic_answer,trust_remote_code=true,max_length=8192 \
      --use-gantry \
      --gantry-args env-secret="OPENAI_API_KEY=openai_api_key" \
      --task-args chat_overrides="{\"generation_kwargs\": {\"stop_sequences\": [\"Problem:\", \"Answer:\", \"Question:\", \"</s>\", \"<|eot_id|>\"]}}" \
      --fim-tokens l2c \
      --oe-eval-branch davidh/olmo3 \
      --beaker-image oe-eval-beaker/oe_eval_olmo3_auto \
      --vllm-use-v1-spec \
      --dashboard olmo3-midtraining-mixing \
      --workspace ai2/oe-data \
      --budget ai2/oe-base


    


    echo "Completed experiment for $exp_id"
  } &
done

wait 
echo "All experiments completed." 


: 'olmo-cookbook-eval evaluate \
        "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step2385-hf" \
        --tasks olmo3:dev:7b:main:v2 \
        --cluster ai2/jupiter-cirrascale-2 \
        --partition-size 8 \
        --num-gpus 1 \
        --model-backend vllm \
        --model-args trust_remote_code=true,max_length=4096 \
        --beaker-image oe-eval-beaker/oe_eval_olmo3_auto\
        --fim-tokens l2c \
        --vllm-use-v1-spec \
        --vllm-memory-utilization 0.7 \
        --dashboard olmo3-midtraining-mixing \
        --workspace ai2/oe-data \
        --budget ai2/oe-base \
        --priority urgent'