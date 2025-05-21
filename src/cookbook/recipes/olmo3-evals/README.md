# Evaluation practices for OLMo 3 development


## In-loop evaluation

For **OLMo 3 7B** integration tests use this set of evals, baked into the OLMo 3 7B config in OLMo-Core: https://github.com/allenai/OLMo-core/blob/a91a82e6b8b37103f738e190cdddd6278f2c7f1f/src/scripts/train/OLMo3-7B.py#L113. In summary:
* Don't slow down training run:
    * Prune to minimal set of tasks + use a fast version of MC. This is captured in OLMo Core as the `fast` set. See PR: https://github.com/allenai/OLMo-core/pull/282
* Focus on BPB + MC, ignoring RC:
    * BPB should spot any major issues early on in training before MC takeoff
    * Still track MC because we want to make sure there is sensible metric takeoff. For an example of OLMo 3 7B MC metric takeoff slightly after 150B tokens, see https://wandb.ai/ai2-llm/olmo3/reports/OLMo-3-vs-OLMo-2--VmlldzoxMjc2MTA4Mw


For **OLMo 2 1B 5xC** or **OLMo 2 7B annealing** runs, which we are still using for data ablations, the broad recommendation is to rely more on offline eval, which always has the latest state of evals.

Some more notes about in-loop eval:
* If you want to add an in-loop eval, the repo is here: https://github.com/allenai/OLMo-in-loop-evals
* When selecting which metrics in Wandb, be very careful around whether you are selecting `{dev|test}`, `{rc|mc}`, `{length-normalized-accuracy|length-normalized-accuracy v2}`, `{BPB|BPB v2}`, `{5shot|5shot_fast}`.


## Offline evaluation

For all OLMo models (+ external baselines), this is a running list of evals we care about.

For **OLMo 2 1B 5xC** runs, it's still good practice to look at both BPB & RC numbers, which usually track together with each other; MC numbers typically haven't broken through noise at this point.

The command to run an eval looks like:
```python
olmo-cookbook-eval evaluate "/oe-training-default/ai2-llm/checkpoints/mayeec/olmo-cookbook-core-v2-1bv2-5xC-dclm-baseline-topic-classified-sample-natural-28f8e9a9/step61000-hf" \
    --tasks arc_easy:rc::olmes \
    --tasks arc_challenge:rc::olmes \
    --tasks hellaswag:rc::olmes \
    --tasks mmlu:rc \
    --tasks basic_skills:rc::olmes \
    --tasks minerva \
    --tasks codex_humaneval:3shot:bpb::none \
    --tasks mbpp:3shot:bpb::none \
    --tasks mt_mbpp \
    --priority high \
    --cluster 80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --partition-size 8 \
    --dashboard olmo-3-evals  \
    --workspace ai2/olmo-3-evals
```

*Task names are collected here: https://github.com/allenai/olmo-cookbook/blob/e20beaee74a6a10b18113520e9e907fdbc24f444/src/cookbook/constants.py#L478*


To pull dashboard results:

```
olmo-cookbook-eval results \
    --dashboard olmo-3-evals \
    --tasks arc_easy:rc::olmes \
    --tasks arc_challenge:rc::olmes \
    --tasks hellaswag:rc::olmes

olmo-cookbook-eval results \
    --dashboard olmo-3-evals \
    --tasks mmlu:rc

olmo-cookbook-eval results \
    --dashboard olmo-3-evals \
    --tasks basic_skills:rc

olmo-cookbook-eval results \
    --dashboard olmo-3-evals \
    --tasks minerva

olmo-cookbook-eval results \
    --dashboard olmo-3-evals \
    --tasks codex_humaneval:3shot:bpb::none \
    --tasks mbpp:3shot:bpb::none

olmo-cookbook-eval results \
    --dashboard olmo-3-evals \
    --tasks mt_mbpp
```

*Notes*
* I don't know why `basic_skills` pull dashboard requires removing `:rc::olmes` but launching eval requires adding it or it'll only launch the Arithmetic subportion. Something weird.

## FAQs

1. **Why leave out GSM8k?** The task is odd and appears mainly moved by hill-climbing with mid-training data. Minerva seems like it covers a greater range.

2. **BPB vs RC vs MC?** This is still debated among team, but the eventual goal should be to move toward BPB that we trust for our experiments, and monitoring MC (or our final target end task format) on 7B+ runs for metric breakthrough moments & final scores.


## TODOs

1. Want to add MMLU subcategories also pulled as part of dashboard pull.
2. Want to add more evals. There are 3 themes:
    * Existing evals that others use, but we don't. Candidates include LAMBADA, NQ, Squad, TriviaQA, MedMCQA, MedQA, etc.
    * Existing evals that need to be fixed in some way. Candidates include converting Gen2MC format, data augmentation for PiQA, SIQA, CSQA, OBQA, Winogrande, Simplified version of hard tasks like SimpleQA, etc.
    * New evals that capture something we aren't evaluating today but we think important to capture. Candidates include legal document tasks, science IE/Summ tasks, structured input (tables) reasoning tasks, perplexity over gold reasoning chains, etc
3. Want to add some stats testing, or some notion of noise.