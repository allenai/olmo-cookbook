import logging
from pathlib import Path
from typing import List, Tuple

import yaml
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerLaunchConfig,
    BeakerWekaBucket,
)

from cookbook.aliases import (
    ExperimentConfig,
    ExperimentGroup,
    ExperimentInstance,
    SourceConfig,
    SourceInstance,
)

logger = logging.getLogger(__name__)


def config_from_path(config: Path) -> ExperimentConfig:
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data, path=config)


def mk_source_instances(
    sources: list[SourceConfig], priors: Tuple[dict[str, float], int] | None = None
) -> list[SourceInstance]:
    # If no user provided ratios, use the priors from the sources
    if priors:
        ratios_by_source, total_tokens = priors
    else:
        # TODO(undfined): Clean this up and fail faster
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
    config: ExperimentConfig, group_uuid: str, priors: Tuple[dict[str, float], int]
) -> list[ExperimentInstance]:
    """Generate source instances from a config."""
    return [
        ExperimentInstance(
            name=f"{config.name}-{group_uuid}",
            sources=mk_source_instances(config.dataset.sources, priors),
        )
    ]


def mk_experiment_group(
    config: ExperimentConfig, priors: Tuple[dict[str, float], int], group_uuid: str
) -> ExperimentGroup:
    """Build an experiment group from an experiment config."""

    return ExperimentGroup(
        config=config,
        group_id=group_uuid,
        instances=mk_experiments(config, group_uuid, priors),
    )


def mk_instance_cmd(
    instance: ExperimentInstance, config: ExperimentConfig, group_id: str, beaker_user: str
) -> List[str]:
    """Build a command for launching an experiment instance."""

    return [
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


def mk_launch_configs(group: ExperimentGroup, beaker_user: str) -> list[BeakerLaunchConfig]:
    """Build a beaker launch config from an experiment group."""

    weka_buckets: List[BeakerWekaBucket] = []
    if group.config.weka:
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))

    return [
        BeakerLaunchConfig(
            name=f"{experiment.name}",
            description=group.config.description,
            task_name=experiment.name,
            cmd=mk_instance_cmd(experiment, group.config, group.group_id, beaker_user),
            clusters=[group.config.cluster],
            num_nodes=group.config.nodes,
            num_gpus=group.config.gpus,
            shared_filesystem=group.config.shared_filesystem,
            allow_dirty=True,
            weka_buckets=weka_buckets,
            budget=group.config.budget or "ai2/oe-data",
            workspace=group.config.workspace,
            preemptible=group.config.preemptible,
            # Add a new cookbook specific image, this will work for now
            beaker_image="ai2-tylerm/olmo-core-nightly",
            priority=group.config.priority,
            env_secrets=[
                BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
                BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
                BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
                BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
                BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
                BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
            ],
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
