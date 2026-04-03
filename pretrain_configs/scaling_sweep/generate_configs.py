#!/usr/bin/env python3
"""Generate YAML configs for the cookbook model-size × compute-budget sweep.

Sweep:
  Model sizes: olmo2_100M, olmo2_190M, olmo2_370M
  Compute budgets: 0.5xC, 1xC, 2xC
  Treatments: baseline, icl_overlap (50/50 mix)
  Total: 18 configs

Uses the paper's LR scaling formula and Chinchilla token budget formula.
"""
import math
from pathlib import Path

OUTDIR = Path(__file__).parent

# Non-embedding params (from TransformerConfig.num_non_embedding_params with dolma2 tokenizer)
MODELS = {
    "olmo2_190M": {"non_emb": 190_354_176, "gpus": 4},
    "olmo2_370M": {"non_emb": 371_262_464, "gpus": 8},
    "olmo2_600M": {"non_emb": 462_466_368, "gpus": 8},
}

BUDGETS = {"0.5xC": 0.5, "1xC": 1.0, "2xC": 2.0}

SEQ_LEN = 2048
TOK_PER_PARAM = 20
DEFAULT_RMBS = 16 * SEQ_LEN  # 32768

BASELINE_PATHS = """\
        - weka://oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy
        - weka://oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy
        - weka://oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy
        - weka://oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/s2pdf_redacted/**/*.npy
        - weka://oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy
        - weka://oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy"""

ICL_PATH = "weka://oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v01/icl-overlap-max-suffix-2048-eos-fix-500B-sample/allenai/dolma2-tokenizer/*.npy"


def compute_config(model_name, model_cfg, budget_label, budget_mult):
    N = model_cfg["non_emb"]
    gpus = model_cfg["gpus"]

    lr = 0.0047 * (N / 108_000_000) ** (-1 / 3)

    # Global batch size: round paper formula to nearest multiple of (gpus * rmbs)
    raw_batch = round(SEQ_LEN * 160 * (N / 108_000_000) ** (2 / 3))
    unit = gpus * DEFAULT_RMBS
    global_batch_size = max(round(raw_batch / unit), 1) * unit

    # Token budget: 20 * N * c, rounded to batch multiple
    raw_tokens = int(TOK_PER_PARAM * N * budget_mult)
    steps = max(round(raw_tokens / global_batch_size), 1)
    max_tokens = steps * global_batch_size

    # Warmup: ~10% of steps for this budget, min 100
    warmup = max(round(steps * 0.10), 100)

    return lr, global_batch_size, max_tokens, steps, warmup


def make_yaml(model_name, model_cfg, budget_label, budget_mult, treatment):
    lr, gbs, max_tokens, steps, warmup = compute_config(
        model_name, model_cfg, budget_label, budget_mult
    )
    gpus = model_cfg["gpus"]
    size_tag = model_name.replace("olmo2_", "")
    budget_tag = budget_label.lower().replace(".", "p")
    treat_tag = "icl-overlap-50-50" if treatment == "icl_overlap" else "baseline"

    name = f"suffix-train-{size_tag}-{budget_tag}-{treat_tag}"
    desc_treat = "ICL overlap 50/50 mix" if treatment == "icl_overlap" else "baseline"

    eval_interval = max(round(steps / 10), 100)
    # Round eval_interval to something clean
    eval_interval = round(eval_interval / 100) * 100
    eval_interval = max(eval_interval, 100)

    rmbs_line = ""
    if gpus >= 4:
        rmbs = 16 * SEQ_LEN
        rmbs_line = f"\nrank_microbatch_size: {rmbs}"

    if treatment == "baseline":
        dataset_block = f"""\
dataset:
  sources:
    - name: dolma2-full
      target_ratio: 1.0
      paths:
{BASELINE_PATHS}"""
    else:
        dataset_block = f"""\
dataset:
  chunk_based_mixture: true
  sources:
    - name: dolma2-full-icl-overlap-suffix-2048-eos-fix-500B
      target_ratio: 0.5
      paths:
        - {ICL_PATH}
    - name: dolma2-100b-baseline
      target_ratio: 0.5
      paths:
{BASELINE_PATHS}"""

    yaml_content = f"""\
name: "{name}"
description: "suffix train {desc_treat} {size_tag} @ {budget_label} on dolma2 — with in-loop evals"
budget: "ai2/oe-base"
workspace: "ai2/dolma2"
nodes: 1
gpus: {gpus}
preemptible: true
max_tokens: {max_tokens}
sequence_length: {SEQ_LEN}
global_batch_size: {gbs}{rmbs_line}
seed: 1337
learning_rate: {lr}
warmup_steps: {warmup}
model: "{model_name}"
tokenizer: "dolma2"
weka: true
priority: high
cluster: ai2/jupiter
lm_evaluator: true
downstream_evaluators:
  - olmo2_dev_1b
eval_interval: {eval_interval}
{dataset_block}
"""
    return name, yaml_content


if __name__ == "__main__":
    all_configs = []
    for model_name, model_cfg in MODELS.items():
        for budget_label, budget_mult in BUDGETS.items():
            for treatment in ["baseline", "icl_overlap"]:
                name, content = make_yaml(
                    model_name, model_cfg, budget_label, budget_mult, treatment
                )
                fname = f"{name}.yaml"
                outpath = OUTDIR / fname
                outpath.write_text(content)
                all_configs.append((name, fname))
                size = model_name.replace("olmo2_", "")
                print(f"  {size:>4s} {budget_label:>4s} {treatment:>12s} -> {fname}")

    print(f"\nGenerated {len(all_configs)} configs in {OUTDIR}")
    print("\nLaunch commands:")
    for name, fname in all_configs:
        print(f"olmo-cookbook launch -c pretrain_configs/scaling_sweep/{fname}")
