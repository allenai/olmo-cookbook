import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import cache, partial
from typing import Any, TypedDict

import platformdirs
import requests

from cookbook.cli.utils import (
    format_datalake_tags,
    make_eval_run_name,
    unpack_datalake_tags,
)


class Experiment(TypedDict):
    experiment_id: str
    author_name: str
    task_name: str
    model_name: str
    tags: dict[str, Any]


class InspectedExperiment(TypedDict):
    ALL_METRICS_files: int
    METRICS_files: int
    PREDICTIONS_files: int
    REQUESTS_files: int
    INPUTS_files: int
    METADATA_files: int


class ResultType(Enum):
    all_metrics = "ALL_METRICS"
    metrics = "METRICS"
    predictions = "PREDICTIONS"
    requests = "REQUESTS"
    inputs = "INPUTS"
    metadata = "METADATA"


OE_EVAL_DATALAKE_BASE_API_URL = "https://oe-eval-datalake.allen.ai"
CACHE_APPLICATION_NAME = "oe-eval-datalake"
CACHE_USER_NAME = "olmo-cookbook"


def get_logger():
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    # Check if the logger already has handlers to avoid duplicates
    if not log.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")  # Simple formatter to just show the message
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)
    return log


logger = get_logger()


def inspect_experiment(
    experiment: Experiment,
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
) -> InspectedExperiment:
    """Inspects an experiment."""
    url = f"{base_url}/greenlake/inspect/{experiment['experiment_id']}"
    response = requests.get(url, params={})
    return InspectedExperiment(**{**experiment, **response.json()})


class TaskMetadataDict(TypedDict):
    alias: str


class _TaskConfigDict(TypedDict, total=False):
    metadata: TaskMetadataDict


class TaskConfigDict(_TaskConfigDict):
    num_shots: int
    primary_metric: str
    split: str
    task_core: str
    task_name: str


class TaskDict(TypedDict):
    task_hash: str
    task_config: TaskConfigDict


@cache
def get_task(task_hash: str, base_url: str = OE_EVAL_DATALAKE_BASE_API_URL) -> TaskDict:
    url = f"{base_url}/bluelake/get-task-config/{task_hash}"
    response = requests.get(url)
    response.raise_for_status()
    return TaskDict(**response.json())


class ModelConfigDict(TypedDict):
    model: str


class ModelDict(TypedDict):
    model_hash: str
    model_config: ModelConfigDict


@cache
def get_model(model_hash: str, base_url: str = OE_EVAL_DATALAKE_BASE_API_URL) -> ModelDict:
    url = f"{base_url}/bluelake/get-model-config/{model_hash}"
    response = requests.get(url)
    response.raise_for_status()
    return ModelDict(**response.json())


def _download_results(
    experiment: Experiment,
    result_type: ResultType,
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
    invalidate_cache: bool = False,
    cache_dir: str | None = None,
    debug: bool = False,
) -> list[dict]:
    """Downloads a result file for an eval experiment."""

    # figure out where to save cache
    cache_dir = cache_dir or platformdirs.user_cache_dir(CACHE_APPLICATION_NAME, CACHE_USER_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{experiment['experiment_id']}-{result_type.name}.jsonl")

    if not invalidate_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            logger.info("Reading cache from %s for experiment %s", cache_path, experiment["experiment_id"])
            return [json.loads(line) for line in f]

    inspected_experiment = inspect_experiment(experiment, base_url=base_url)
    if (count_files := inspected_experiment[f"{result_type.value}_files"]) == 0:  # pyright: ignore
        return []

    def download_single(task_idx: int) -> list[dict[str, Any]]:
        url = f"{base_url}/greenlake/download-result/{experiment['experiment_id']}"
        params = {"resulttype": result_type.value, "task_idx": task_idx}
        response = requests.get(url, params=params)
        response.raise_for_status()

        results: list[dict[str, Any]] = []
        for line in response.iter_lines(decode_unicode=True):
            data: dict[str, Any] = json.loads(line)
            task_config = get_task(data["task_hash"], base_url=base_url)
            model_config = get_model(data["model_hash"], base_url=base_url)
            assert "metadata" in task_config["task_config"], "metadata not found in task_config"
            task_alias = task_config["task_config"]["metadata"]["alias"]
            model_alias = model_config["model_config"]["model"]
            results.append({**data, "task_name": task_alias, "model_name": model_alias})

        logger.info(
            "Downloaded %d results for experiment %s, task_idx %d",
            len(results),
            experiment["experiment_id"],
            task_idx,
        )
        return results

    start = time.time()
    max_workers = 1 if debug else count_files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [
            task_result
            for task_group in executor.map(download_single, range(count_files))
            for task_result in task_group
        ]
    delta = time.time() - start
    logger.info(
        "Downloaded %d results using %d workers for experiment %s in %.2f seconds",
        len(results),
        max_workers,
        experiment["experiment_id"],
        delta,
    )

    with open(cache_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
        logger.info("Wrote cache to %s for experiment %s", cache_path, experiment["experiment_id"])

    return results


class PredictionDict(TypedDict):
    doc_id: int
    metrics: dict[str, float]
    model_output: list
    task_hash: str
    mode_hash: str
    task_name: str
    model_name: str


def download_predictions(
    experiment: Experiment,
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
    invalidate_cache: bool = False,
    cache_dir: str | None = None,
    debug: bool = False,
) -> list[PredictionDict]:
    """Downloads predictions for an experiment."""
    results = _download_results(
        experiment=experiment,
        result_type=ResultType.predictions,
        base_url=base_url,
        invalidate_cache=invalidate_cache,
        cache_dir=cache_dir,
        debug=debug,
    )
    predictions = [PredictionDict(**row) for row in results]
    logger.info("Downloaded %d predictions for experiment %s", len(predictions), experiment["experiment_id"])
    return predictions


class _MetricsValuesDict(TypedDict, total=False):
    acc_raw: float
    acc_per_token: float
    acc_per_char: float
    acc_per_byte: float
    sum_logits_corr: float
    logits_per_token_corr: float
    logits_per_char_corr: float
    bits_per_byte_corr: float
    extra_metrics: dict[str, float]


class MetricsValuesDict(_MetricsValuesDict):
    primary_score: float


class MetricsDict(TypedDict):
    task_name: str
    model_name: str
    task_hash: str
    model_hash: str
    metrics: MetricsValuesDict
    model_config: ModelConfigDict
    task_config: TaskConfigDict
    task_idx: int


def download_metrics(
    experiment: Experiment,
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
    invalidate_cache: bool = False,
    cache_dir: str | None = None,
    debug: bool = False,
) -> list[MetricsDict]:
    """Downloads metrics for an experiment."""
    result = _download_results(
        experiment=experiment,
        result_type=ResultType.metrics,
        base_url=base_url,
        invalidate_cache=invalidate_cache,
        cache_dir=cache_dir,
        debug=debug,
    )
    metrics = [MetricsDict(**row) for row in result]
    logger.info("Downloaded %d metrics for experiment %s", len(metrics), experiment["experiment_id"])
    return metrics


def _find_experiments(
    from_created_dt: str = "2024-07-01",
    limit: int | None = 10_000,
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
    datalake_tags: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> list[Experiment]:
    """Finds experiments."""
    url = f"{base_url}/bluelake/find-experiments/"

    params = {
        "from_created_dt": from_created_dt,
        "limit": limit,
        "return_fields": "experiment_id,author_name,task_name,model_name,tags",
    }
    if datalake_tags:
        params["tags"] = format_datalake_tags(datalake_tags)

    if model_name:
        params["model_name"] = model_name

    response = requests.get(url, params=params)
    response.raise_for_status()
    experiments = []
    for row in response.json():
        tags = unpack_datalake_tags(row.pop("tags"))
        experiments.append(Experiment(**row, tags=tags))
    return experiments


def find_experiments_by_checkpoint_path(
    checkpoint_path: str,
    add_bos_token: bool = False,
    from_created_dt: str = "2024-07-01",
    limit: int | None = 10_000,
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
) -> list[Experiment]:
    """Finds experiments associated with a checkpoint path."""

    # we first try to see if model_path yields to any results by specifying it as tag
    experiments = _find_experiments(
        from_created_dt=from_created_dt,
        limit=limit,
        base_url=base_url,
        datalake_tags={"model_path": checkpoint_path},
    )

    logger.info(f"Found {len(experiments)} experiments for model path {checkpoint_path} (tags method)")

    if len(experiments) > 0:
        return experiments

    model_name = make_eval_run_name(checkpoint_path=checkpoint_path, add_bos_token=add_bos_token)
    experiments = _find_experiments(
        from_created_dt=from_created_dt,
        limit=limit,
        base_url=base_url,
        model_name=model_name,
    )

    logger.info(f"Found {len(experiments)} experiments for model name {model_name}")
    return experiments


def find_experiment_by_dashboard_name(
    dashboard_name: str,
    from_created_dt: str = "2024-07-01",
    limit: int | None = 10_000,
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
) -> list[Experiment]:
    """Finds experiments associated with a dashboard name."""

    experiments = _find_experiments(
        from_created_dt=from_created_dt,
        limit=limit,
        base_url=base_url,
        datalake_tags={"dashboard": dashboard_name},
    )
    logger.info(f"Found {len(experiments)} experiments for dashboard name {dashboard_name}")

    return experiments


def _process_fine_grained_predictions(
    experiments: list[Experiment],
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
    invalidate_cache: bool = False,
    cache_dir: str | None = None,
    debug: bool = False,
) -> tuple[list[PredictionDict], list[MetricsDict]]:
    def get_experiment(
        experiment: Experiment,
        _base_url: str = base_url,
        _invalidate_cache: bool = invalidate_cache,
        _cache_dir: str | None = cache_dir,
        _debug: bool = debug,
    ) -> tuple[list[PredictionDict], list[MetricsDict]]:
        predictions = download_predictions(
            experiment=experiment,
            base_url=_base_url,
            invalidate_cache=_invalidate_cache,
            cache_dir=_cache_dir,
            debug=_debug,
        )
        metrics = download_metrics(
            experiment=experiment,
            base_url=_base_url,
            invalidate_cache=_invalidate_cache,
            cache_dir=_cache_dir,
            debug=_debug,
        )
        return predictions, metrics

    max_workers = 1 if debug else len(experiments)
    logger.info("Downloading results for %d experiments with %d workers", len(experiments), max_workers)

    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        predictions, metrics = map(
            lambda grouped_experiments: sum(grouped_experiments, []),
            zip(*executor.map(get_experiment, experiments)),
        )
    delta = time.time() - start

    logger.info("Found %d predictions and %d metrics in %.2f seconds", len(predictions), len(metrics), delta)
    return predictions, metrics


def get_experiment_fine_grained_predictions(
    checkpoint_path: str,
    add_bos_token: bool = False,
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
    invalidate_cache: bool = False,
    cache_dir: str | None = None,
    debug: bool = False,
) -> tuple[list[PredictionDict], list[MetricsDict]]:
    """Gets fine-grained results for an experiment."""
    experiments = find_experiments_by_checkpoint_path(
        checkpoint_path=checkpoint_path,
        add_bos_token=add_bos_token,
        base_url=base_url,
    )

    logger.info(f"Found {len(experiments)} experiments for checkpoint path {checkpoint_path}")

    return _process_fine_grained_predictions(
        experiments=experiments,
        base_url=base_url,
        invalidate_cache=invalidate_cache,
        cache_dir=cache_dir,
        debug=debug,
    )


def get_dashboard_fine_grained_predictions(
    dashboard_name: str,
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
    invalidate_cache: bool = False,
    cache_dir: str | None = None,
    debug: bool = False,
) -> tuple[list[PredictionDict], list[MetricsDict]]:
    experiments = find_experiment_by_dashboard_name(
        dashboard_name=dashboard_name,
        base_url=base_url,
    )

    logger.info(f"Found {len(experiments)} experiments for dashboard name {dashboard_name}")

    return _process_fine_grained_predictions(
        experiments=experiments,
        base_url=base_url,
        invalidate_cache=invalidate_cache,
        cache_dir=cache_dir,
        debug=debug,
    )


def get_simple_dashboard_metrics(
    dashboard_name: str,
    base_url: str = OE_EVAL_DATALAKE_BASE_API_URL,
    invalidate_cache: bool = False,
    cache_dir: str | None = None,
    debug: bool = False,
):
    experiments = find_experiment_by_dashboard_name(
        dashboard_name=dashboard_name,
        base_url=base_url,
    )

    logger.info(f"Found {len(experiments)} experiments for dashboard name {dashboard_name}")

    max_workers = 1 if debug else len(experiments)
    logger.info("Downloading results for %d experiments with %d workers", len(experiments), max_workers)

    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fn = partial(
            download_metrics,
            base_url=base_url,
            invalidate_cache=invalidate_cache,
            cache_dir=cache_dir,
            debug=debug,
        )
        metrics = sum(executor.map(fn, experiments), [])
    delta = time.time() - start

    logger.info(
        "Found %d metrics for %d experiments in %.2f seconds in dashboard %s",
        len(metrics),
        len(experiments),
        delta,
        dashboard_name,
    )

    return metrics
