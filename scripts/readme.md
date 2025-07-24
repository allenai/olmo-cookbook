```bash scripts/run_task_baseline_sampler.sh "*olmo3:dev:7b:vllm" ianm-7b-baseline
olmo-cookbook-eval results --dashboard ianm-7b-baseline --tasks olmo3:dev:7b:mini --format json > baseline_7B_sampler_data.json
python scripts/baseline_task_viz.py baseline_7B_sampler_data.json --outdir baseline_7b_task_viz
```

```
olmo-cookbook-eval results --dashboard ianm-1b-baseline --tasks olmo3:dev:1b --format json > baseline_sampler_data.json
```