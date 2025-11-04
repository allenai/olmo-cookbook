import logging
import math
import os
from pathlib import Path
from typing import List, Tuple, Union, cast, Optional
from urllib.parse import urlparse

import gcsfs
import s3fs
import yaml
from olmo_core.io import normalize_path
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
)
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback
from olmo_core.utils import seed_all

from cookbook.aliases import (
    ExperimentConfig,
    ExperimentGroup,
    ExperimentInstance,
    SourceConfig,
    SourceInstance,
)
from cookbook.model.builder import TransformerConfigBuilder
from cookbook.utils.data import normalize_source_paths

logger = logging.getLogger(__name__)


def config_from_path(config: Path) -> ExperimentConfig:
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data, path=config)


def mk_source_instances(
    sources: list[SourceConfig], priors: Tuple[dict[str, float], int] | None = None
) -> list[SourceInstance]:
    if priors:
        ratios_by_source, total_tokens = priors
    else:
        ratios_by_source = {}

    instances = []
    for source in sources:
        ratio = source.target_ratio or ratios_by_source[source.name]
        instances.append(
            SourceInstance(
                name=source.name,
                paths=source.paths,
                ratio=ratio,
                repetition_factor=source.repetition_factor,
            )
        )

    return instances


def mk_experiments(
    config: ExperimentConfig, group_id: str, priors: Tuple[dict[str, float], int]
) -> list[ExperimentInstance]:
    """Generate source instances from a config."""
    return [
        ExperimentInstance(
            name=f"{config.name}-{group_id}",
            sources=mk_source_instances(config.dataset.sources, priors),
        )
    ]


def mk_experiment_group(
    config: ExperimentConfig, priors: Tuple[dict[str, float], int], group_id: str
) -> ExperimentGroup:
    """Build an experiment group from an experiment config."""

    return ExperimentGroup(
        config=config,
        group_id=group_id,
        instances=mk_experiments(config, group_id, priors),
    )


def mk_swarm_source_instances(
    sources: list[SourceConfig], mix_map: dict[str, tuple[float, float]]
) -> list[SourceInstance]:
    instances = []

    for source in sources:
        if source.topics:
            for topic in source.topics:
                full_name = f"{source.name}:{topic.name}"
                if full_name not in mix_map or mix_map[full_name][0] == 0:
                    continue
                instances.append(
                    SourceInstance(
                        name=full_name,
                        paths=topic.paths,
                        ratio=mix_map[full_name][0],
                        repetition_factor=mix_map[full_name][1],
                    )
                )
        else:
            if source.name not in mix_map or mix_map[source.name][0] == 0:
                continue
            instances.append(
                SourceInstance(
                    name=source.name,
                    paths=source.paths,
                    ratio=mix_map[source.name][0],
                    repetition_factor=mix_map[source.name][1],
                )
            )

    return instances

def mk_swarm_experiments(
    config: ExperimentConfig, mixes: list[dict[str, tuple[float, float]]], group_id: str
) -> list[ExperimentInstance]:
    """Generate source instances from a config."""
    return [
        ExperimentInstance(
            name=f"{config.name}-{group_id}-{idx:02}",
            sources=mk_swarm_source_instances(config.dataset.sources, mix),
        )
        for idx, mix in enumerate(mixes)
    ]

def mk_swarm_experiment_group(
    config: ExperimentConfig, mixes: list[dict[str, tuple[float, float]]], group_id: str
) -> ExperimentGroup:
    """Build an experiment group from an experiment config."""

    return ExperimentGroup(
        config=config,
        group_id=group_id,
        instances=mk_swarm_experiments(config, mixes, group_id),
    )

def mk_instance_cmd(
    instance: ExperimentInstance, config: ExperimentConfig, group_id: str, beaker_user: str, swarm: bool
) -> List[str]:
    """Build a command for launching an experiment instance."""

    cmd = [
        "src/cookbook/train.py",
        "train",
        "-n",
        instance.name,
        "-g",
        group_id,
        "-u",
        beaker_user,
        "-C",
        str(config.path),
    ]

    if swarm:
        sources = []
        for source in instance.sources:
            paths = [f'"{path}"' for path in source.paths]
            source_str = (
                f'-s ("{source.name}",[{",".join(paths)}],{source.ratio},{source.repetition_factor})'
            )
            sources.append(source_str)
        cmd.extend(sources)

    return cmd


_REMOTE_FS_CACHE: dict[str, Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]] | None = None


def remote_fs_cache() -> dict[str, Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]]:
    global _REMOTE_FS_CACHE
    if _REMOTE_FS_CACHE is not None:
        return _REMOTE_FS_CACHE

    _REMOTE_FS_CACHE = dict(
        s3=s3fs.S3FileSystem(),
        weka=s3fs.S3FileSystem(client_kwargs={"endpoint_url": os.environ["WEKA_ENDPOINT_URL"]}, profile="WEKA"),
        gs=gcsfs.GCSFileSystem(),
    )

    return _REMOTE_FS_CACHE


def build_train_config(config_path: Path, run_name: str, group_id: str, beaker_user: str, dry_run: bool = False, source: List[Tuple[str, List[str], str, str]] = []):
    """
    Launch a training run with the given parameters.
    """

    base_config = config_from_path(config_path)
    load_path_fs = None

    if dry_run:
        source_paths = base_config.dataset.sources
        if base_config.load_path:
            try:
                load_path_fs = remote_fs_cache()[urlparse(base_config.load_path).scheme]
            except KeyError:
                raise ValueError(f"Unsupported load path scheme: {base_config.load_path}")

            # When we have a weka path locally we need to treat it like a remote s3
            # path and strip the special weka prefix and bucket name
            base_config.load_path = normalize_path(base_config.load_path.replace("weka://", "s3://"))

    else:
        if base_config.load_path:
            # When we have a weka path remotely on beaker we need to treat it like a local path since the bucket is mounted
            base_config.load_path = normalize_path(base_config.load_path.replace("weka://", "/weka/"))

    if len(source) > 0:
        source_instances: List[SourceInstance] = []
        for item in source:
            name, paths, ratio, repetition = item
            source_instances.append(
                SourceInstance(
                    name=name, paths=paths, ratio=float(ratio), repetition_factor=float(repetition)
                )
            )
    else:
        source_paths = normalize_source_paths(base_config.dataset.sources, expand=True)
        source_instances = mk_source_instances(source_paths, None)
    dp_world_size = base_config.nodes * base_config.gpus

    config = TransformerConfigBuilder(
        beaker_user=beaker_user,
        cluster=base_config.cluster,
        downstream_evaluators=base_config.downstream_evaluators,
        dtype=base_config.dataset.dtype,
        eval_interval=base_config.eval_interval,
        group_id=group_id.strip(),
        lm_evaluator=base_config.lm_evaluator,
        max_dp_world_size=dp_world_size,
        max_target_sequence_length=base_config.max_target_sequence_length,
        max_tokens=base_config.max_tokens,
        model_identifier=base_config.model,
        run_name=run_name.strip(),
        save_interval=base_config.save_interval,
        seed=base_config.seed,
        sequence_length=base_config.sequence_length,
        sources=source_instances,
        tokenizer=base_config.tokenizer,
        metrics_config=base_config.metrics_config,
        weka=base_config.weka,
        rank_microbatch_size=base_config.rank_microbatch_size,
        global_batch_size=base_config.global_batch_size,
        load_path=base_config.load_path,
        warmup_steps=base_config.warmup_steps,
        learning_rate=base_config.learning_rate,
        scheduler_type=base_config.scheduler_type,
        annealing=base_config.annealing,
        hard_stop=base_config.hard_stop,
        model_overrides=base_config.model_overrides,
        activation_checkpointing=base_config.activation_checkpointing,
        load_path_fs=load_path_fs,
    ).build()

    seed_all(config.init_seed)
    config_dict = config.as_config_dict()
    trainer = None

    if not dry_run:
        dataset = config.dataset.build()
        model = config.model.build(init_device="meta")
        train_module = config.train_module.build(model)
        data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = config.trainer.build(train_module, data_loader)

        # If we have a load path and there is no checkpoint in the save folder, load the checkpoint from the load path.
        if not trainer.maybe_load_checkpoint(trainer.save_folder) and base_config.load_path:
            logger.info(
                f"Loading checkpoint from {base_config.load_path} and load_trainer_state: {base_config.load_state}"
            )
            trainer.load_checkpoint(base_config.load_path, load_trainer_state=base_config.load_state, load_optim_state=base_config.load_optim_state)

        cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    logger.info("Configuration:")
    # We log estimated step count here when dry_run is enabled because we're not able to build the trainer on non-CUDA devices
    if dry_run:
        logger.info(
            f"Estimated training steps: {math.ceil(base_config.max_tokens / config.data_loader.global_batch_size):,}"
        )
    logger.info(config)

    return trainer


def mk_launch_configs(group: ExperimentGroup, beaker_user: str, swarm: bool=False) -> list[BeakerLaunchConfig]:
    """Build a beaker launch config from an experiment group."""

    weka_buckets: List[BeakerWekaBucket] = []
    if group.config.weka:
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))

    return [
        BeakerLaunchConfig(
            name=f"{experiment.name}",
            description=group.config.description,
            task_name=experiment.name,
            cmd=mk_instance_cmd(experiment, group.config, group.group_id, beaker_user, swarm),
            clusters=[group.config.cluster],
            num_nodes=group.config.nodes,
            num_gpus=group.config.gpus,
            shared_filesystem=group.config.weka,
            allow_dirty=True,
            weka_buckets=weka_buckets,
            budget=group.config.budget or "ai2/oe-base",
            workspace=group.config.workspace,
            preemptible=group.config.preemptible,
            beaker_image="petew/olmo-core-tch270cu128",
            priority=group.config.priority,
            env_vars=[BeakerEnvVar(name="NCCL_DEBUG", value="INFO" if group.config.nccl_debug else "WARN")],
            env_secrets=[
                BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
                BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
                BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
                BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
                BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
                BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
                BeakerEnvSecret(name="GOOGLE_CLOUD_PROJECT", secret="GOOGLE_CLOUD_PROJECT"),
            ],
            retries=3,
            setup_steps=[
                'git clone "$REPO_URL"',
                "conda shell.bash activate base",
                "cd olmo-cookbook",
                'git checkout "$GIT_REF"',
                "git submodule update --init --recursive",
                "pip install -e '.[all]'",
                "pip freeze",
                # Move AWS credentials from env to relevant files
                "mkdir -p ~/.aws",
                "printenv AWS_CONFIG > ~/.aws/config",
                "printenv AWS_CREDENTIALS > ~/.aws/credentials",
            ],
        )
        for experiment in group.instances
    ]
