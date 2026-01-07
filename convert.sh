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

    olmo-cookbook-eval convert "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step2385" \
    -t olmo-core-v2 \
    --use-beaker \
    --olmo-core-v2-commit-hash 57a04d0b69047d797c96eede056a211e75b5914a \
    --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
    --huggingface-transformers-commit-hash ae3889ced6ed7362e5883671fc6dc4cb4fece5fa \
    --beaker-allow-dirty

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

    olmo-cookbook-eval convert "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step2385" \
    -t olmo-core-v2 \
    --use-beaker \
    --olmo-core-v2-commit-hash 57a04d0b69047d797c96eede056a211e75b5914a \
    --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
    --huggingface-transformers-commit-hash ae3889ced6ed7362e5883671fc6dc4cb4fece5fa \
    --beaker-allow-dirty

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
    olmo-cookbook-eval convert "/oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step2385" \
      -t olmo-core-v2 \
      --use-beaker \
      --olmo-core-v2-commit-hash 57a04d0b69047d797c96eede056a211e75b5914a \
      --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
      --huggingface-transformers-commit-hash ae3889ced6ed7362e5883671fc6dc4cb4fece5fa \
      --beaker-allow-dirty
    echo "Completed experiment for $exp_id"
  } &
done

wait 
echo "All experiments completed." 