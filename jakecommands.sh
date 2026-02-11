python -m cookbook.remote gs://ai2-llm/checkpoints/jakep/olmo3-microanneal-s2orc-natural-0e3d8891/step4769 \
          weka://oe-training-default/ai2-llm/checkpoints/jakepolmo3-microanneal-s2orc-natural-0e3d8891/step4769


python -m cookbook.remote gs://ai2-llm/checkpoints/jakep/olmo3-microanneal-pdfquality-bottom50-pstar-48a1e5e0/step4769 \
          weka://oe-training-default/ai2-llm/checkpoints/jakep/olmo3-microanneal-pdfquality-bottom50-pstar-48a1e5e0/step4769


olmo-cookbook-eval convert "/oe-training-default/ai2-llm/checkpoints/jakep/olmo3-microanneal-pdfquality-rawgptmini-78b23817/step4769" \
 -t olmo-core-v2 \
 --use-beaker \
 --olmo-core-v2-commit-hash 57a04d0b69047d797c96eede056a211e75b5914a \
 --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
 --huggingface-transformers-commit-hash ae3889ced6ed7362e5883671fc6dc4cb4fece5fa


olmo-cookbook-eval convert "/oe-training-default/ai2-llm/checkpoints/jakep/olmo3-microanneal-pdfquality-nofilter-47a2adb5/step4769" \
 -t olmo-core-v2 \
 --use-beaker \
 --olmo-core-v2-commit-hash 57a04d0b69047d797c96eede056a211e75b5914a \
 --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
 --huggingface-transformers-commit-hash ae3889ced6ed7362e5883671fc6dc4cb4fece5fa


olmo-cookbook-eval evaluate \
  "/oe-training-default/ai2-llm/checkpoints/jakep/olmo3-microanneal-pdfquality-rawgptmini-78b23817/step4769-hf" \
  --tasks olmo3:dev:midtrain:v1 \
  --priority high \
  --cluster aus80g \
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
  --dashboard olmo3-jakep-pdf-quality \
  --workspace ai2/oe-data

olmo-cookbook-eval evaluate \
  "/oe-training-default/ai2-llm/checkpoints/jakep/olmo3-microanneal-pdfquality-rawgptmini-78b23817/step4769-hf" \
  --tasks olmo3:dev:7b:main \
  --priority high \
  --cluster aus80g \
  --partition-size 8 \
  --num-gpus 1 \
  --model-backend vllm \
  --model-args trust_remote_code=true,max_length=4096 \
  --beaker-image oe-eval-beaker/oe_eval_olmo3_auto \
  --fim-tokens l2c \
  --vllm-use-v1-spec \
  --dashboard olmo3-jakep-pdf-quality \
  --workspace ai2/oe-data
