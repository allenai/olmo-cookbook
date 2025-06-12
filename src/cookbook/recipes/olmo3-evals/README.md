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

```bash
CHECKPOINT="/oe-training-default/ai2-llm/checkpoints/mayeec/olmo-cookbook-core-v2-1bv2-5xC-dclm-baseline-topic-classified-sample-natural-28f8e9a9/step61000-hf"
CLUSTER="l40"
NUM_GPUS=1
PARTITION=8
PRIORITY="high"
MODEL_ARGS="dtype=bfloat16"
DASHBOARD="olmo-3-evals"
WORKSPACE="ai2/olmo-3-evals"

olmo-cookbook-eval evaluate "$CHECKPOINT" \
    --tasks "*olmo3:dev:1b:vllm" \
    --priority "$PRIORITY" \
    --cluster "$CLUSTER" \
    --num-gpus "$NUM_GPUS" \
    --model-backend vllm \
    --model-args "$MODEL_ARGS" \
    --partition-size "$PARTITION" \
    --dashboard "$DASHBOARD"  \
    --workspace "$WORKSPACE"

olmo-cookbook-eval evaluate "$CHECKPOINT" \
    --tasks "*olmo3:dev:1b:hf" \
    --priority "$PRIORITY" \
    --cluster "$CLUSTER" \
    --num-gpus "$NUM_GPUS" \
    --model-backend hf \
    --model-args "$MODEL_ARGS" \
    --partition-size "$PARTITION" \
    --dashboard "$DASHBOARD"  \
    --workspace "$WORKSPACE"
```

Notes: 
* Task names are collected here: https://github.com/allenai/olmo-cookbook/blob/e20beaee74a6a10b18113520e9e907fdbc24f444/src/cookbook/constants.py*


*How long does it take?*
* "*olmo3:dev:1b:vllm" are a full suite of 20 tasks, each task w multiple metrics + some tasks as families w multiple subtasks. In total, this is around 150 metrics. Takes 2 hours to do all of them on `--partition-size 8` and `num-gpus 1` with single L40 (launches 5 jobs).
* "*olmo3:dev:1b:hf" are two masked PPL evals. Takes 1 hour to do both on a single L40.

To pull dashboard results (use `--format json` to see full results):

```python
olmo-cookbook-eval results \
    --dashboard olmo-3-evals \
    --tasks olmo3:dev:1b \
    --format json | jq '.' | less
```


*Notes*
* If you want to see if the datalake uploading job ran, use your beaker experimental ID to this URL: `https://oe-eval-datalake.allen.ai/greenlake/metadata/01JWMGNY3G3R5N91NW9TCKF6FB`.
* I hate this design, but currently in order to use a named group when pulling dashboard results, you need to use it with a `*`, like `--tasks *mmlu:rc`, which then matches to `ALL_NAMED_GROUPS` in `constants.py`. If you don't include the `*`, it'll pass the exact string `mmlu:rc` and fail to match things.
* I don't know why `basic_skills` pull dashboard requires removing `:rc::olmes` but launching eval requires adding it or it'll only launch the Arithmetic subportion. Something weird.

## FAQs

1. **Why leave out GSM8k?** The task is odd and appears mainly moved by hill-climbing with mid-training data. Minerva seems like it covers a greater range.

2. **BPB vs RC vs MC?** This is still debated among team, but the eventual goal should be to move toward BPB that we trust for our experiments, and monitoring MC (or our final target end task format) on 7B+ runs for metric breakthrough moments & final scores.

### RC vs. MC

For 7B+ runs, our development evals replace multiple-choice RC evals with MC evals. 

Summarizing our conversation with the team, we considered three options for calculating both, and aggregating the results:

1. Calculate `max(rc, mc)` -- This isn't desirable. Imagine two ablations -- one consistently has a very high RC and the other a very high MC. This is not a behavior we want from our metric
2. Calculate `avg(rc, mc)` -- This isn't desirable. At the small scale (when models get random-chance MC) we are artificially penalizing performance. At the large scale, (when models get lower RC than MC, because RC does not allow the model to see distractor options and therefore is a slightly more difficult task config) we are artificially penalizing performance.
3. Keep `mc` only -- We choose this. There is agreement that MC is a better task format, and that the issues caused by aggregation are not worth the benefit of accounting for two task formats.

Additionally, we observed empircally that the MC tasks better ranked models w.r.t. training compute. For more discussion, see [Figure 1 of the OLMES paper](https://arxiv.org/pdf/2406.08446?page=7).

## TODOs

1. Want to add MMLU subcategories also pulled as part of dashboard pull.
2. Want to add more evals. There are 3 themes:
    * Existing evals that others use, but we don't. Candidates include LAMBADA, NQ, Squad, TriviaQA, MedMCQA, MedQA, etc.
    * Existing evals that need to be fixed in some way. Candidates include converting Gen2MC format, data augmentation for PiQA, SIQA, CSQA, OBQA, Winogrande, Simplified version of hard tasks like SimpleQA, etc.
    * New evals that capture something we aren't evaluating today but we think important to capture. Candidates include legal document tasks, science IE/Summ tasks, structured input (tables) reasoning tasks, perplexity over gold reasoning chains, etc
3. Want to add some stats testing, or some notion of noise.
