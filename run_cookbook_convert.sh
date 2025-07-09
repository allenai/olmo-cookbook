#path format /oe-training-default/ai2-llm/.../step1234

olmo-cookbook-eval convert ${1} \
-t olmo-core-v2 \
--use-beaker \
--olmo-core-v2-commit-hash 013ef7b54aa2d583f9811ec6211a536da407a4b1 \
--huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
--huggingface-transformers-commit-hash ca728b8879ce5127ea3e2f8d309c2c5febab5dc5
