import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields as dataclass_fields
from datetime import datetime
from functools import partial
from sys import stderr
from threading import current_thread, main_thread
from typing import Callable, ClassVar, Generic, List, TypeVar

import requests
from tqdm import tqdm
from typing_extensions import Self

from cookbook.eval.cache import get_datalake_cache

T = TypeVar("T")
V = TypeVar("V")


@dataclass
class BaseDatalakeItem(Generic[T]):
    _base_url: ClassVar[str] = "https://oe-eval-datalake.allen.ai"
    _from_created_dt: ClassVar[str] = "2024-07-01"

    def __post_init__(self):
        for field in dataclass_fields(self):
            # we check if a field has a parser function and use to initialize it
            if (parser := field.metadata.get("parser", None)) is not None:
                setattr(self, field.name, parser(getattr(self, field.name)))

    @staticmethod
    def _thread_number() -> int:
        if current_thread() == main_thread():
            return 0
        else:
            return int(current_thread().name.split("-")[-1])

    @classmethod
    def run(cls, **kwargs: T) -> List[Self]:
        """Run a query to retrieve one or more datalake items of this type."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def _prun(
        cls,
        fns: list[Callable[[], V]],
        num_workers: int | None = None,
        quiet: bool = False,
        position: int = 0,
    ) -> list[V]:
        # Validate input arguments
        if len(fns) == 0:
            return []

        results: list[V] = []
        with ExitStack() as stack:
            # Set up thread pool and progress bar
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=num_workers))
            pbar = stack.enter_context(
                tqdm(total=len(fns), desc=cls.__name__, disable=quiet, file=stderr, position=position)
            )

            # Submit all tasks to the thread pool
            futures = [pool.submit(fn) for fn in fns]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    # Cancel all remaining tasks if any task fails
                    for remaining_future in futures:
                        remaining_future.cancel()
                    raise e
                pbar.update(1)

        return results

    @classmethod
    def prun(cls, num_workers: int | None = None, quiet: bool = False, **kwargs: list[T]) -> list[Self]:
        """Same as run() method, but runs in parallel.

        Args:
            num_workers: Number of workers to use. If None, the number of workers
                will be the default in ThreadPoolExecutor.
            **kwargs: Keyword arguments to pass to the run() method.

        Returns:
            List of datalake items.
        """
        num_args = -1
        for k, v in kwargs.items():
            assert isinstance(v, list), f"Argument {k} must be a list"
            num_args = len(v) if num_args == -1 else num_args
            assert len(v) == num_args, f"Argument {k} must have the same length as the previous arguments"
        assert num_args > 0, "At least one argument must be provided"

        fns = [partial(cls.run, **{k: v[i] for k, v in kwargs.items()}) for i in range(num_args)]

        # actually run the function in parallel
        results = cls._prun(fns=fns, num_workers=num_workers, quiet=quiet)
        return [result for result_group in results for result in result_group]


@dataclass
class Tag:
    """
    A tag is a key-value pair; it is returned as a string like "key=value,key2=value2".
    """

    key: str
    value: str

    def __post_init__(self):
        assert "=" not in self.key and "=" not in self.value, "Key and value cannot contain '=' character"
        assert "," not in self.key and "," not in self.value, "Key and value cannot contain ',' character"

    def __str__(self) -> str:
        return f"{self.key}={self.value}" if self.key != self.value else self.key

    @classmethod
    def from_str(cls, s: str | list[Self]) -> list[Self]:
        if isinstance(s, list):
            # already parsed
            return s

        # make the tags
        return [cls(*elem.split("=", 1)) if "=" in elem else cls(elem, elem) for elem in s.strip().split(",")]


@dataclass
class FindExperiments(BaseDatalakeItem):
    """Find all experiments for a given dashboard."""

    experiment_id: str
    model_name: str
    task_name: str
    created: datetime = dataclass_field(metadata=dict(parser=lambda x: datetime.fromisoformat(x)))

    # parser argument will make sure that nested dataclasses are initialized
    tags: list[Tag] = dataclass_field(default_factory=list, metadata=dict(parser=Tag.from_str))

    _endpoint: ClassVar[str] = "bluelake/find-experiments/"

    @classmethod
    def run(cls, dashboard: str | None = None, model_name: str | None = None, limit: int = 10_000) -> list[Self]:
        # make sure at least one of dashboard or model_name is provided
        assert dashboard or model_name, "Either dashboard or model_name must be provided"
        response = requests.get(
            f"{cls._base_url}/{cls._endpoint}",
            params={
                "from_created_dt": cls._from_created_dt,
                **({"tags": f"dashboard={dashboard}"} if dashboard else {}),
                **({"model_name": model_name} if model_name else {}),
                "limit": limit,
                "return_fields": ",".join(f.name for f in dataclass_fields(cls)),
            },
            headers={"accept": "application/json"},
        )

        response.raise_for_status()
        all_records = [cls(**experiment) for experiment in response.json()]

        # Sort records by created date (newest first)
        all_records.sort(key=lambda x: x.created, reverse=True)

        return all_records


@dataclass
class Metrics:
    """
    A collection of values for a given task and model.
    """

    primary_score: float
    logits_per_byte_corr: float | None = None
    bits_per_byte_corr: float | None = None

    # extra metrics are task-specific and are sometimes returned as a dictionary
    extra_metrics: dict = dataclass_field(default_factory=dict)

    @property
    def bpb(self) -> float | None:
        return self.bits_per_byte_corr or self.logits_per_byte_corr

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        # these are the names of the fields that are shared across tasks
        fields = {f.name for f in dataclass_fields(cls)}

        # move all metrics that are not shared across tasks to extra_metrics
        extra_metrics = {**d.pop("extra_metrics", {}), **{k: d.pop(k) for k in list(d) if k not in fields}}
        return cls(**d, extra_metrics=extra_metrics)


@dataclass
class MetricsAll(BaseDatalakeItem):
    """Find all metrics for a given experiment."""

    compute_config: dict
    current_date: str
    metrics: Metrics = dataclass_field(metadata=dict(parser=Metrics.from_dict))
    model_config: dict
    model_hash: str
    num_instances: int
    processing_time: float
    task_config: dict
    task_hash: str
    task_idx: int
    task_name: str

    # this is not a field in dataclass
    _endpoint: ClassVar[str] = "greenlake/metrics-all/"

    @property
    def alias(self) -> str:
        """
        The alias used to identify the task when launching experiments.
        """
        task_alias = self.task_config.get("metadata", {}).get("alias", None)

        if task_alias is None:
            # we hash the task_config in json format and get the first 6 characters as suffix
            task_suffix = hashlib.sha256(json.dumps(self.task_config).encode()).hexdigest()[:6]
            task_alias = f"{self.task_name}-{task_suffix}"

        return task_alias

    @property
    def model_path(self) -> str | None:
        return self.model_config.get("model_path", None)

    @property
    def model_name(self) -> str | None:
        model_name = self.model_config.get("model", None)
        if "revision" in self.model_config and self.model_config["revision"] is not None:
            model_name = model_name + ":" + self.model_config["revision"]
        return model_name

    @property
    def is_aggregate(self) -> bool:
        return self.task_config.get("metadata", {}).get("num_tasks", 0) > 0

    @classmethod
    def run(cls, experiment_id: str, force: bool = False, skip_on_fail: bool = False) -> List[Self]:
        cache = get_datalake_cache()
        if not (result := cache.get(experiment_id=experiment_id)).success or force:
            response = requests.get(
                f"{cls._base_url}/{cls._endpoint.rstrip('/')}/{experiment_id}",
                headers={"accept": "application/json"},
            )
            try:
                response.raise_for_status()
            except Exception as e:
                if skip_on_fail:
                    return []
                else:
                    raise e

            result = cache.set(response.json(), experiment_id=experiment_id)

        result = [cls(**metric) for metric in (result.value or [])]

        return result


@dataclass
class RemoveFromDashboard(BaseDatalakeItem):
    """Remove an experiment from a dashboard."""

    _endpoint: ClassVar[str] = "bluelake/add-experiment-tags/"

    @classmethod
    def _endpoint_request(cls, experiment_id: str, tags: list[Tag], overwrite: bool = True) -> Self:
        """
        Making a request to update the tags of an experiment; override=True replaces all tags,
        while override=False adds the new tags without removing the existing ones.
        """

        params = {
            "tags": ",".join([str(tag) for tag in tags]),
            "overwrite": overwrite,
            "experiment_id": experiment_id,
        }
        response = requests.put(
            f"{cls._base_url}/{cls._endpoint}",
            headers={"accept": "application/json"},
            params=params,
        )
        response.raise_for_status()
        return cls(**(response.json() or {}))

    @classmethod
    def run(cls, model_name: str, dashboard: str, fuzzy: bool = False) -> List[Self]:
        runs = FindExperiments.run(model_name=model_name, dashboard=dashboard)
        cache = get_datalake_cache()

        fns = []
        for run in runs:
            # if the experiment is in the cache, we remove it since we changed its tags
            cache.delete(experiment_id=run.experiment_id)

            if not fuzzy and run.model_name != model_name:
                continue

            new_tags = [tag for tag in run.tags if (tag.key != "dashboard" or tag.value != dashboard)]
            fn = partial(cls._endpoint_request, experiment_id=run.experiment_id, tags=new_tags, overwrite=True)
            fns.append(fn)

        print(f"Removing {len(runs):,} experiments from dashboard {dashboard}")

        if (n := cls._thread_number()) > 0:
            # if we are already running in parallel, then we can use _prun() to create more parallelism
            # (we avoid making a second progress bar by setting quiet=True)
            return cls._prun(fns=fns, num_workers=len(runs), position=n)
        else:
            # if we are single-threaded, then we can just run the function sequentially
            return [fn() for fn in fns]


@dataclass
class AddToDashboard(RemoveFromDashboard):
    """Add an experiment to a dashboard."""

    @classmethod
    def run(cls, model_name: str, dashboard: str, fuzzy: bool = False) -> List[Self]:
        runs = FindExperiments.run(model_name=model_name)
        cache = get_datalake_cache()
        fns = []
        for run in runs:
            if not fuzzy and run.model_name != model_name:
                continue

            # if the experiment is in the cache, we remove it since we changed its tags
            cache.delete(experiment_id=run.experiment_id)

            # we first check if the dashboard tag is already present
            if any(tag.key == "dashboard" and tag.value == dashboard for tag in run.tags):
                continue

            # if it is not present, then we add it
            new_tags = [Tag(key="dashboard", value=dashboard)]
            fn = partial(cls._endpoint_request, experiment_id=run.experiment_id, tags=new_tags, overwrite=False)
            fns.append(fn)

        print(f"Adding {len(runs):,} experiments to dashboard {dashboard}")

        if (n := cls._thread_number()) > 0:
            # if we are already running in parallel, then we can use _prun() to create more parallelism
            # (we avoid making a second progress bar by setting quiet=True)
            return cls._prun(fns=fns, num_workers=len(runs), position=n)
        else:
            # if we are single-threaded, then we can just run the function sequentially
            return [fn() for fn in fns]
