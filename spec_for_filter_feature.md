# A feature to return results recalculated over a subset of item level results.

Currently a command like `olmo-cookbook-eval results --dashboard ianm-decon-olmo3 --tasks olmo3:dev:7b:main:v1` returns just the aggregate metric over all items in each benchmark. We would like to be able to add a feature to provide a file of (`doc_ids` int , `task_names` list of str) tuples that represent items to filter, and then the returned results will be calculated over the filtered item level results based on the raw prediction files.

## getting the data

Currently the `class MetricsAll(BaseDatalakeItem):` handles getting metrics and this makes a request to the datalake at "greenlake/metrics-all/". Instead we want to get predictions data from the datalake. This will require a few requests. We get the experiment ID the same we currently do for metrics-all.

First we need to know how many predictions files are in the experiment with a request like:
```
curl -X 'GET' \
  'https://oe-eval-datalake.allen.ai/greenlake/inspect/01K1GWB92WZX1T1YPGYQZNFP6X?resulttype=PREDICTIONS' \
  -H 'accept: application/json'
```
Then for each of the files (for however many of them there are we itterate task_idx) we make a request like this
```
curl -X 'GET' \
  'https://oe-eval-datalake.allen.ai/greenlake/get-result/01K1GWB92WZX1T1YPGYQZNFP6X?resulttype=PREDICTIONS&task_idx=0' \
  -H 'accept: application/json'
```

This gives us data like:
```
{
  "doc_id": 0,
  "native_id": "Mercury_7175875",
  "metrics": {
    "predicted_index_raw": 2,
    "predicted_index_per_token": 2,
    "predicted_index_per_char": 2,
    "predicted_index_per_byte": 2,
    "predicted_index_uncond": null,
    "correct_choice": 2,
    "acc_raw": 1,
    "acc_per_token": 1,
    "acc_per_char": 1,
    "acc_per_byte": 1,
    "acc_uncond": null,
    "no_answer": 0,
    "sum_logits_corr": -0.09184452891349792,
    "logits_per_token_corr": -0.09184452891349792,
    "logits_per_char_corr": -0.04592226445674896,
    "bits_per_byte_corr": 0.06625182319819083
  },
  ...
```

We also need some information from the metric.jsonl files (again itterateing the task_idx)

```
curl -X 'GET' \
  'https://oe-eval-datalake.allen.ai/greenlake/get-result/01K1GWB92WZX1T1YPGYQZNFP6X?resulttype=METRICS&task_idx=0' \
  -H 'accept: application/json'
```

which looks like this
```
{
  "task_name": "arc_challenge:mc",
  "task_hash": "4f74c98974b297e7e9dd6a0c1a76ce80",
  "model_hash": "39ee62684648536901a8aee91abee216",
  "model_config": {
    "model": "decon-anneal-round3-webround2-100B-olmo3_7b_with-reasoning-anneal-12T-fc803782_step47684-hf",
    "revision": null,
    "trust_remote_code": true,
    "max_length": 4096,
    "model_path": "weka://oe-training-default/ai2-llm/checkpoints/ianm/decon-anneal-round3-webround2-100B-olmo3_7b_with-reasoning-anneal-12T-fc803782/step47684-hf",
    "model_type": "vllm",
    "add_bos_token": false,
    "gpu_memory_utilization": 0.7,
    "vllm_for_mc": true
  },
  "task_config": {
    "task_name": "arc_challenge:mc",
    "task_core": "arc_challenge",
    "limit": null,
    "split": "all",
    "num_shots": 5,
    "fewshot_seed": 1234,
    "primary_metric": "acc_raw",
```

## reaggregating

Now we know that `"primary_metric": "acc_raw"` and `"task_name": "arc_challenge:mc"`. So we can find all tuples in our filter input that have this task_name in their task_names, and then select just the entries for the doc_ids in those tuples. Finally we grab the "metrics" field corresponding to the primary_metric for each prediction and we simply take the average of that value over all filtered items.

These results will be output in the same format as the MerticsAll class run method and this new class will also support all the other methods that MetricsAll does (this may need to also make a metrics all requrest to the datalake) so that everything can be interopperable with existing code that uses MetricsAll objects. The only difference is that run will return metrics reaggregated over just the filtered items.

## plumbing

Lastly we need to plumb through a couple flags for the `olmo-cookbook-eval results`.

We will have `--filter_exclude <path to tuple file>` which will cause the code to use the new class in place of MetricsAll and to read the tuples in from the provided file path. This will exclude any task items that match the tuples from the aggregate number.
We will also have `--filter_include <path to tuple file>` which does the same thing but instead only includes task items that match the tuples and excludes all other items from the aggregate.

## reporting

Finally when the user is using the default format please include a report of the count of the items matched for each task name as well as the total number of items in the task.