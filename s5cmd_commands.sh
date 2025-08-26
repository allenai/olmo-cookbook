set -e
s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/midtraining-reasoning/OpenMathReasoning/OpenMathReasoning-rewrite-full-thoughts/jsonls-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/midtraining-reasoning/OpenMathReasoning/OpenMathReasoning-rewrite-full-thoughts/jsonls-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/midtraining-reasoning/tinyMATH/MIND/data/processed-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/MIND/data/processed-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/midtraining-reasoning/tinyMATH/PoT/processed_data/processed-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/midtraining-reasoning/flat_dolmino_math-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/midtraining-reasoning/flat_dolmino_math-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/finemath/finemath-3plus-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/finemath/finemath-3plus-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/reddit-rewrites/densesub_highthresh_microanneal_4omini_rewrite/documents-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/reddit-rewrites/densesub_highthresh_microanneal_4omini_rewrite/documents-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/reddit-rewrites/densesub_lowthresh_4omini_rewrite/documents-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/reddit-rewrites/densesub_lowthresh_4omini_rewrite/documents-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/thinking-data/big-reasoning-traces-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/thinking-data/big-reasoning-traces-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/thinking-data/qwq-traces-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/thinking-data/qwq-traces-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/thinking-data/s1k-gemini-traces-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/thinking-data/s1k-gemini-traces-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/midtraining-reasoning/math-meta-reasoning/documents-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/midtraining-reasoning/math-meta-reasoning/documents-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/midtraining-reasoning/code-meta-reasoning/documents-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/midtraining-reasoning/code-meta-reasoning/documents-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/midtraining-reasoning/verifiable/gpt-41/documents-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/midtraining-reasoning/verifiable/gpt-41/documents-decon-2/allenai/dolma2-tokenizer/"

s5cmd cp -sp "/mnt/raid0/ai2-llm/pretraining-data/sources/midtraining-reasoning/openmathreasoning/teacher-student-lecture/documents-decon-2/documents-decon-2/allenai/dolma2-tokenizer/" "s3://ai2-llm/preprocessed/midtraining-reasoning/openmathreasoning/teacher-student-lecture/documents-decon-2/documents-decon-2/allenai/dolma2-tokenizer/"

