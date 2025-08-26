set -e
s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/tulu-3-sft-for-olmo-3-midtraining/raw-decon-2/tulu-3-midtrain-v0-data-simple-concat-with-new-line-with-generation-prompt/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/tulu-3-sft-for-olmo-3-midtraining/raw-decon-2/tulu-3-midtrain-v0-data-simple-concat-with-new-line-with-generation-prompt/allenai/dolma2-tokenizer/"

