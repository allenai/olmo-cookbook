
name=akshitab/OLMo2-7B-public-mix/step238419
#name=kevinfarhat/OLMo2-7B-anneal-freeze-mj-finemath4plus/step11921
#name=kevinfarhat/OLMo2-7B-anneal-b3/step11921
#name=kevinfarhat/OLMo2-7B-anneal-lbv1/step11921	


olmo-cookbook-eval convert \
	"/oe-training-default/ai2-llm/checkpoints/$name" \
	-t olmo-core --use-beaker --huggingface-tokenizer allenai/OLMo-2-1124-7B		
	
