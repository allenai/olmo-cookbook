import logging
from pathlib import Path
from typing import List, Tuple, cast

import yaml

from cookbook.aliases import ExperimentConfig, ExperimentGroup, ExperimentInstance, SourceConfig, SourceInstance
from cookbook.model.builder import TransformerConfigBuilder
from cookbook.utils.data import normalize_source_paths
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerLaunchConfig, BeakerWekaBucket
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback
from olmo_core.utils import get_default_device, seed_all

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


def build_train_config(config_path: Path, run_name: str, group_id: str, beaker_user: str, dry_run: bool = False):
    """
    Launch a training run with the given parameters.
    """

    base_config = config_from_path(config_path)

    # Because this is happening on-box in Beaker we want paths normalized for usage there.
    source_instances = mk_source_instances(normalize_source_paths(base_config.dataset.sources), None)
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
        wandb_config=base_config.wandb,
        weka=base_config.weka,
    ).build()

    device = get_default_device()
    world_mesh = config.model.build_mesh(device=device)

    seed_all(config.init_seed)
    config_dict = config.as_config_dict()

    trainer = None
    if not dry_run:
        model = config.model.build(
            init_device="meta",
            device=device,
            mesh=world_mesh,
            max_seq_len=config.dataset.sequence_length,
        )
        dataset = config.dataset.build()
        optim = config.optim.build(model)
        data_loader = config.data_loader.build(dataset=dataset, mesh=world_mesh)
        trainer = config.trainer.build(model, optim, data_loader, mesh=world_mesh)
        cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    logger.info("Configuration:")
    logger.info(config_dict)

    return trainer


def validate_experiment_group(group: ExperimentGroup) -> bool:
    """Stack all the checks for validity of the ExperimentGroup here.
    Return True if everything is A-okay
    """
    checks_passing = True
    msgs = []
    # Check: if running on augusta => not using weka for anything
    if "augusta" in group.config.cluster and group.config.weka:
        checks_passing &= False
        msgs.append("Cannot be on Augusta and use weka!")

    return checks_passing, msgs


def mk_launch_configs(group: ExperimentGroup, beaker_user: str) -> list[BeakerLaunchConfig]:
    """Build a beaker launch config from an experiment group."""

    try:
        exp_group_valid, validity_msgs = validate_experiment_group(group)
        assert exp_group_valid
    except AssertionError as e:
        logger.info("Exp group not valid!")
        for msg in validity_msgs:
            logger.info("\t" + msg)
        raise e

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
            shared_filesystem=group.config.weka,
            allow_dirty=True,
            weka_buckets=weka_buckets,
            budget=group.config.budget or "ai2/oe-data",
            workspace=group.config.workspace,
            preemptible=group.config.preemptible,
            beaker_image="petew/olmo-core-tch260cu124",
            priority=group.config.priority,
            env_secrets=[
                BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
                BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
                BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
                BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
                BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
                BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
                BeakerEnvSecret(name="GS_INTEROP_KEY", secret="GS_INTEROP_KEY"),
                BeakerEnvSecret(name="GS_INTEROP_SECRET", secret="GS_INTEROP_SECRET"),
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
