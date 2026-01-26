import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gcsfs
import olmo_core.train.train_module as train_module
import s3fs
import torch
from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.io import resource_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    CosWithWarmup,
    OptimConfig,
    OptimGroupOverride,
    Scheduler,
    SkipStepAdamWConfig,
)
from olmo_core.optim.scheduler import WSD, CosWithWarmupAndLinearDecay, LinearWithWarmup
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    BeakerCallback,
    Callback,
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
)

from cookbook.aliases import (
    AnnealConfig,
    MetricBackend,
    MetricsConfig,
    SchedulerType,
    SourceInstance,
)
from cookbook.cli.core import estimate_batch_size
from cookbook.data.dataset import MixtureBuilder
from cookbook.model.config import (
    DEFAULT_LR_MAP,
    ModelConfigIdentifier,
    ModelTrainConfig,
    Tokenizers,
    WrappedTransformerConfig,
)
from cookbook.model.evaluators import DownstreamEvaluator, get_tasks_for_groups

logger = logging.getLogger(__name__)


@dataclass
class SchedulerState:
    global_step: int
    max_steps: int
    last_pretrain_step: int
    max_pretrain_steps: int
    base_lr: float
    starting_lr: float


@dataclass
class TransformerConfigBuilder:
    """
    A builder class for configuring and creating a transformer model training configuration.

    Attributes:
        run_name (str): The name of the run.
        sources (List[SourceInstance]): A list of source instances.
        sequence_length (int): The sequence length for the model.
        max_target_sequence_length (int): The maximum target sequence length for the model.
        max_tokens (int): The maximum number of tokens to train on.
        model_identifier (ModelConfigIdentifier): The identifier for the model.
        transformer_config (TransformerConfig): The transformer configuration.
        group_id (str): The group ID for the run.
        cluster (str): The cluster name.
        beaker_user (str): The Beaker user name.
        s3 (bool): Whether to use S3 for storage.
        seed (int): The random seed for reproducibility.
        tokenizer (TokenizerConfig): The tokenizer configuration.
        dtype (str): The data type for the dataset.
        weka (bool): Whether to use Weka buckets.
        metrics_config (Optional[MetricsConfig]): The metrics configuration, if any.
        max_dp_world_size (int): The maximum data parallel world size.
        eval_interval (int): The evaluation interval.
        save_interval (int): The save interval.
        lm_evaluator (bool): Whether to enable language model evaluation.
        downstream_evaluators (List[DownstreamEvaluator]): The downstream evaluators.
        load_path (Optional[str]): Path to load a model checkpoint from.
        hard_stop (Optional[Duration]): The hard stop duration.
        learning_rate (Optional[float]): The learning rate for the optimizer.
        global_batch_size (Optional[int]): The global batch size.
        rank_microbatch_size (Optional[int]): The rank microbatch size.
        warmup_steps (Optional[int]): The number of warmup steps for the scheduler.
        scheduler_type (SchedulerType): The type of scheduler to use. Default is SchedulerType.COS_LINEAR.
        model_overrides (Optional[List[str]]): Optional dotlist overrides for the model configuration.
        activation_checkpointing (bool): Whether to enable activation checkpointing.
        profile (bool): Whether to enable profiling.

    Methods:
        __init__(run_name, sources, sequence_length, max_tokens, group_id, cluster, beaker_user,
                 tokenizer, dtype, model_identifier, weka, max_dp_world_size, save_interval, eval_interval,
                 lm_evaluator, downstream_evaluators, load_path=None, global_batch_size=None,
                 rank_microbatch_size=None, learning_rate=None, metrics_config=None,
                 max_target_sequence_length=8192, seed=42, s3=True, profile=False):
            Initializes the TransformerConfigBuilder.

        get_tokenizer_config(tokenizer: str) -> TokenizerConfig:
            Returns the tokenizer configuration based on the tokenizer identifier.

        get_warmup_steps() -> int:
            Returns the number of warmup steps.

        get_global_batch_size() -> int:
            Returns the global batch size based on the sequence length or user-defined value.

        get_rank_microbatch_size() -> int:
            Returns the rank microbatch size for training.

        get_learning_rate() -> float:
            Returns the learning rate for the optimizer.

        next_power_of_2(x: int) -> int:
            Returns the next power of 2 greater than or equal to x.

        build_callbacks() -> Dict[str, Callback]:
            Builds and returns a dictionary of callbacks for the trainer.

        build_dataset_config() -> NumpyDatasetConfig:
            Builds and returns the dataset configuration.

        get_scheduler_config(scheduler_type: SchedulerType) -> Scheduler:
            Returns the scheduler configuration based on the scheduler type.

        get_optimizer_config() -> OptimConfig:
            Returns the optimizer configuration.

        build() -> ModelTrainConfig:
            Builds and returns the model training configuration.
    """

    run_name: str
    sources: List[SourceInstance]
    sequence_length: int
    max_target_sequence_length: int
    max_tokens: int
    model_identifier: ModelConfigIdentifier
    transformer_config: TransformerConfig
    group_id: str
    cluster: str
    beaker_user: str
    s3: bool
    seed: int
    tokenizer: TokenizerConfig
    dtype: str
    weka: bool
    metrics_config: Optional[MetricsConfig]
    max_dp_world_size: int
    eval_interval: int
    save_interval: int
    lm_evaluator: bool
    cluster: str
    downstream_evaluators: List[DownstreamEvaluator]  # type: ignore
    scheduler_type: SchedulerType
    model_overrides: Optional[List[str]]
    hard_stop: Optional[Duration]
    load_path: Optional[str]
    learning_rate: Optional[float]
    global_batch_size: Optional[int]
    rank_microbatch_size: Optional[int]
    warmup_steps: Optional[int]
    load_path_fs: Optional[Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]]
    activation_checkpointing: bool
    annealing: Optional[AnnealConfig] = None
    profile: bool = False
    chunk_based_mixture: bool = False

    def __init__(
        self,
        run_name: str,
        sources: List[SourceInstance],
        sequence_length: int,
        max_tokens: int,
        group_id: str,
        cluster: str,
        beaker_user: str,
        tokenizer: str,
        dtype: str,
        model_identifier: ModelConfigIdentifier,
        weka: bool,
        max_dp_world_size: int,
        save_interval: int,
        eval_interval: int,
        lm_evaluator: bool,
        downstream_evaluators: List[DownstreamEvaluator],  # type: ignore
        scheduler_type: SchedulerType,
        activation_checkpointing: bool = False,
        model_overrides: Optional[List[str]] = None,
        load_path_fs: Optional[Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]] = None,
        annealing: Optional[AnnealConfig] = None,
        hard_stop: Optional[Duration] = None,
        load_path: Optional[str] = None,
        global_batch_size: Optional[int] = None,
        rank_microbatch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        metrics_config: Optional[MetricsConfig] = None,
        max_target_sequence_length: int = 8192,
        seed: int = 42,
        warmup_steps: Optional[int] = None,
        profile: bool = False,
        chunk_based_mixture: bool = False,
    ):
        self.run_name = run_name
        self.sources = sources
        self.sequence_length = sequence_length
        self.max_tokens = max_tokens
        self.group_id = group_id
        self.seed = seed
        self.model_identifier = model_identifier
        self.tokenizer = self.get_tokenizer_config(tokenizer=tokenizer)
        self.model_overrides = model_overrides
        self.transformer_config = WrappedTransformerConfig.from_model_identifier(model_identifier, self.tokenizer)
        self.beaker_user = beaker_user.strip()
        self.profile = profile
        self.activation_checkpointing = activation_checkpointing
        self.data_dir = "s3://ai2-llm"
        self.dataset_dtype = NumpyDatasetDType[dtype]
        self.root_dir = f"/tmp/{self.run_name}"
        self.metrics_config = metrics_config
        self.max_dp_world_size = max_dp_world_size
        self.max_target_sequence_length = max_target_sequence_length
        self.max_grad_norm = 1.0
        self.save_interval = save_interval
        self.dataset_detype = NumpyDatasetDType[dtype]
        self.lm_evaluator = lm_evaluator
        self.learning_rate = learning_rate
        self.global_batch_size = global_batch_size
        self.rank_microbatch_size = rank_microbatch_size
        self.downstream_evaluators = downstream_evaluators
        self.warmup_steps = warmup_steps
        self.load_path = load_path
        self.hard_stop = hard_stop
        self.annealing = annealing
        self.load_path_fs = load_path_fs
        self.scheduler_type = scheduler_type
        self.checkpoint_dir = f"{self.data_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"
        self.eval_interval = eval_interval
        self.cluster = cluster
        self.chunk_based_mixture = chunk_based_mixture

        if any(substring in cluster for substring in ["augusta"]):
            self.root_dir = "gs://ai2-llm"
            self.checkpoint_dir = f"{self.root_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"
            # NOTE: work_dir must be a local path, not a url
            self.work_dir = f"/tmp/{self.beaker_user.lower()}/{self.run_name}/dataset-cache"

        elif (
            any(substring in cluster for substring in ["jupiter", "saturn", "ceres", "neptune", "titan"]) and weka
        ):
            self.root_dir = "/weka/oe-training-default/ai2-llm"
            logger.info(f"Using Weka bucket as root dir: {self.root_dir}")
            self.checkpoint_dir = f"{self.root_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"
            self.work_dir = f"{self.root_dir}/{self.beaker_user.lower()}/{self.run_name}/dataset-cache"

        else:
            self.work_dir = f"{self.root_dir}/{self.beaker_user.lower()}/{self.run_name}/dataset-cache"

    def get_tokenizer_config(self, tokenizer) -> TokenizerConfig:
        try:
            return Tokenizers[tokenizer].value
        except ValueError as e:
            logger.info(f"Invalid tokenizer identifier: {tokenizer}")
            raise e

    def get_warmup_steps(self) -> int:
        if not self.warmup_steps == None:
            logger.info(f"Using user-defined warmup steps: {self.warmup_steps}")
            return self.warmup_steps

        return 2000

    def get_global_batch_size(self) -> int:
        if self.global_batch_size:
            global_batch_size = self.global_batch_size
        else:
            global_batch_size = (
                estimate_batch_size(sequence_length=self.sequence_length, total_tokens=self.max_tokens)
                * self.sequence_length
            )

        print(f"Global batch size (in tokens) is: {global_batch_size}")

        return global_batch_size

    def get_rank_microbatch_size(self) -> int:
        if self.rank_microbatch_size is not None:
            rbz = self.rank_microbatch_size
            logger.info(f"Rank microbatch size (in tokens) is: {rbz}")
        else:
            if self.sequence_length == 2048:
                rbz = 16 * self.sequence_length
            elif self.sequence_length == 4096:
                rbz = 8 * self.sequence_length
            else:
                rbz = 4 * self.sequence_length

            logger.info(
                f"Using default rank microbatch size: {rbz} for sequence length: {self.sequence_length}, if OOM errors occur try reducing rank microbatch size"
            )

        return rbz

    def get_learning_rate(self) -> float:
        if self.learning_rate is not None:
            lr = self.learning_rate
        else:
            lr = DEFAULT_LR_MAP.get(self.model_identifier.value, 5e-4)

        return lr

    def next_power_of_2(self, x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def build_callbacks(self) -> Dict[str, Callback]:
        callbacks = {
            "checkpointer": CheckpointerCallback(
                save_interval=self.save_interval,
                ephemeral_save_interval=100,
                save_async=True,
            ),
            "config_saver": ConfigSaverCallback(),
            "profiler": ProfilerCallback(enabled=self.profile),
            "garbage_collector": GarbageCollectorCallback(),
        }

        if self.beaker_user is not None:
            callbacks["beaker"] = BeakerCallback()

        if torch.cuda.is_available():
            callbacks["gpu_monitor"] = GPUMemoryMonitorCallback()

        if self.metrics_config:
            if MetricBackend.wandb in self.metrics_config.backends:
                if self.metrics_config.workspace != MetricsConfig().workspace:
                    # show warning if workspace is set to non-default value;
                    # it is ignored for wandb metrics, only entity is used
                    # (it is used for comet metrics)
                    logger.warning(
                        "metrics_config.workspace is ignored for WandB metrics. Use metrics_config.entity instead."
                    )

                callbacks[MetricBackend.wandb.value] = WandBCallback(
                    name=self.run_name.strip(),
                    project=self.metrics_config.project.strip(),
                    entity=self.metrics_config.entity.strip(),
                    group=self.group_id.strip(),
                    cancel_check_interval=10,
                    enabled=True,
                )
            if MetricBackend.comet in self.metrics_config.backends:
                if self.metrics_config.entity != MetricsConfig().entity:
                    # show warning if entity is set to non-default value;
                    # it is not used for comet metrics (only workspace is used)
                    logger.warning(
                        "metrics_config.entity is ignored for Comet metrics. Use metrics_config.workspace instead."
                    )

                callbacks[MetricBackend.comet.value] = CometCallback(
                    name=self.run_name.strip(),
                    workspace=self.metrics_config.workspace.strip(),
                    project=self.metrics_config.project.strip(),
                    enabled=True,
                    cancel_check_interval=10,
                )

        if self.lm_evaluator:
            callbacks["lm_evaluator"] = LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    name=NumpyDatasetType.padded_fsl,
                    mix_base_dir=self.root_dir,
                    sequence_length=self.sequence_length,
                    tokenizer=self.tokenizer,
                    work_dir=self.work_dir,
                ),
                eval_interval=self.eval_interval,
            )

        if self.downstream_evaluators:
            evaluators = DownstreamEvaluatorCallbackConfig(
                tasks=get_tasks_for_groups(self.downstream_evaluators),
                tokenizer=self.tokenizer,
                eval_interval=self.eval_interval,
            )

            callbacks["downstream_evaluators"] = evaluators

        return callbacks

    def build_dataset_config(self, loader_processes: int = 16) -> NumpyDatasetConfig:
        is_fractional = any(source.ratio is not None and source.ratio != 1 for source in self.sources)

        mixture_config = None
        source_paths = None

        if is_fractional:
            logger.info(
                "Using fractional source_mixture dataset builder... This can take awhile for large token populations!"
            )

            mixture_config = MixtureBuilder(
                sources=self.sources,
                max_tokens=self.max_tokens,
                sequence_length=self.sequence_length,
                seed=self.seed,
                processes=loader_processes,
                dtype=self.dataset_dtype,
            ).build()
        else:
            source_paths = []
            for source in self.sources:
                source_paths.extend(source.paths)

        dataset_config = NumpyDatasetConfig(
            paths=source_paths,
            source_mixture_config=mixture_config,
            name=NumpyDatasetType.fsl,
            sequence_length=self.sequence_length,
            max_target_sequence_length=self.max_target_sequence_length,
            tokenizer=self.tokenizer,
            mix_base_dir=self.root_dir,
            work_dir=self.work_dir,
            chunk_based_mixture=self.chunk_based_mixture,
        )

        return dataset_config

    def get_scheduler_config(self) -> Scheduler:
        scheduler_map = {
            SchedulerType.COSINE: lambda: CosWithWarmup(warmup_steps=self.get_warmup_steps()),
            SchedulerType.COS_LINEAR: lambda: CosWithWarmupAndLinearDecay(
                warmup_steps=self.get_warmup_steps(),
            ),
            SchedulerType.LINEAR: lambda: LinearWithWarmup(
                warmup_steps=self.get_warmup_steps(), alpha_f=0.0 if self.annealing is not None else 0.1
            ),
            SchedulerType.WSD: lambda: WSD(warmup=self.get_warmup_steps()),
        }

        return scheduler_map[self.scheduler_type]()

    def get_optimizer_config(self) -> OptimConfig:
        lr = self.get_learning_rate()

        if self.annealing is not None:
            lr = getattr(self.annealing, "initial_lr", None) or self.get_state_from_checkpoint().starting_lr

        return SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))],
        )

    def get_ac_config(self):
        # NOTE: This is pretty broad, we can make this more fine-grained if we find it useful
        return TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )

    def load_state_and_config_from_path(self) -> Tuple[Path, Path]:
        if not self.load_path:
            raise ValueError(
                "load_path is not set. Please provide a valid load path when attempting to load scheduler state."
            )

        local_cache = f"/tmp/{self.run_name}/"

        if self.load_path_fs:
            train_config = "config.json"
            train_state = "train/rank0.pt"
            logger.info(f"Downloading train state and config from {self.load_path} to {local_cache}")

            for item in [train_state, train_config]:
                self.load_path_fs.download(rpath=f"{self.load_path}/{item}", lpath=local_cache, recursive=True)

            return (
                resource_path(folder=local_cache, fname="rank0.pt"),
                resource_path(folder=local_cache, fname="config.json"),
            )
        else:
            return (
                resource_path(folder=f"{self.load_path}/train", fname="rank0.pt"),
                resource_path(folder=self.load_path, fname="config.json"),
            )

    def get_state_from_checkpoint(self) -> SchedulerState:
        state_path, config_path = self.load_state_and_config_from_path()
        train_state = torch.load(state_path, weights_only=False)

        last_pretrain_step: int = train_state["global_step"]
        max_pretrain_steps: Optional[int] = train_state.get("max_steps", None)

        if max_pretrain_steps is None:
            raise ValueError(
                "Could not find max_steps. Please ensure the checkpoint is valid. Unable to load scheduler state. Exiting!"
            )

        logger.info(f"Will anneal from {last_pretrain_step:,d} of {max_pretrain_steps:,d} total steps")

        if not self.load_path:
            raise ValueError(
                "load_path is not set. Please provide a valid load path when attempting to load scheduler state. Exiting!"
            )

        with open(config_path, "r") as f:
            config = json.load(f)

        del config["dataset"]
        logger.info("Inferring scheduler config from:")
        logger.info(config)

        try:
            # Try olmo_core v2 config format first
            base_lr: int = config["optim"]["lr"]
            scheduler_config = config["train_module"]["scheduler"]
        except KeyError as e:
            # Now try olmo_core v1 config format
            try:
                base_lr: int = config["optim"]["lr"]
                scheduler_config = config["trainer"]["callbacks"]["lr_scheduler"]["scheduler"]
            except KeyError as e:
                logger.error(
                    "Could not find base_lr or scheduler config in train state. Please ensure the checkpoint is valid. Unable to load scheduler state."
                )
                raise e

        scheduler_class = scheduler_config.pop("_CLASS_").split(".")[-1]

        try:
            assert scheduler_class == CosWithWarmup.__name__
        except AssertionError as e:
            logger.error(
                f"Expected scheduler class {CosWithWarmup.__name__}, but got {scheduler_class}: Anneals from a base LR can only be inferred from CosWithWarmup scheduler."
            )
            raise e

        scheduler = CosWithWarmup(**scheduler_config)
        starting_lr = float(scheduler.get_lr(base_lr, last_pretrain_step, max_pretrain_steps))

        return SchedulerState(
            global_step=last_pretrain_step,
            max_steps=max_pretrain_steps,
            last_pretrain_step=last_pretrain_step,
            max_pretrain_steps=max_pretrain_steps,
            base_lr=base_lr,
            starting_lr=starting_lr,
        )

    def build(self) -> ModelTrainConfig:
        global_batch_size = self.get_global_batch_size()
        rank_microbatch_size = self.get_rank_microbatch_size()
        dataset_config = self.build_dataset_config()
        optim_config = self.get_optimizer_config()

        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=global_batch_size,
            work_dir=self.work_dir,
            seed=self.seed,
            num_workers=12,
        )

        load_path = self.load_path
        load_strategy = LoadStrategy.always if load_path else LoadStrategy.if_available

        train_module_config = train_module.TransformerTrainModuleConfig(
            rank_microbatch_size=rank_microbatch_size,
            max_sequence_length=self.sequence_length,
            optim=self.get_optimizer_config(),
            compile_model=True,
            dp_config=train_module.TransformerDataParallelConfig(
                name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
            ),
            ac_config=self.get_ac_config() if self.activation_checkpointing else None,
            float8_config=Float8Config(enabled=False),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
            scheduler=self.get_scheduler_config(),
        )

        trainer_config = TrainerConfig(
            hard_stop=self.hard_stop,
            load_path=load_path,
            load_strategy=load_strategy,
            save_folder=self.checkpoint_dir,
            work_dir=self.work_dir,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=5,
            max_duration=Duration.tokens(self.max_tokens),
        )

        for callback_name, callback in self.build_callbacks().items():
            trainer_config.callbacks[callback_name] = callback

        # Merge any custom dotlist style overrides to the transformer config
        if self.model_overrides:
            logger.info("Applying model overrides:")
            logger.info(self.model_overrides)

            self.transformer_config = self.transformer_config.merge(dotlist=self.model_overrides)

        return ModelTrainConfig(
            init_seed=self.seed,
            model=self.transformer_config,
            optim=optim_config,
            dataset=dataset_config,
            data_loader=data_loader_config,
            trainer=trainer_config,
            train_module=train_module_config,
        )
