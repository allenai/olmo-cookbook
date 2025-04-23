import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import olmo_core.train.train_module as train_module
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
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    CosWithWarmup,
    OptimConfig,
    OptimGroupOverride,
    Scheduler,
    SkipStepAdamWConfig,
)
from olmo_core.optim.scheduler import CosWithWarmupAndLinearDecay
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
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

from cookbook.aliases import MetricBackend, MetricsConfig, SchedulerType, SourceInstance
from cookbook.cli.core import estimate_batch_size
from cookbook.data.dataset import MixtureBuilder
from cookbook.model.config import (
    DEFAULT_LR_MAP,
    ModelConfigIdentifier,
    ModelTrainConfig,
    Tokenizers,
    WrappedTransformerConfig,
)
from cookbook.model.evaluators import DownstreamEvaluator
from cookbook.model.schedulers import WSD

logger = logging.getLogger(__name__)


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
    downstream_evaluators: List[DownstreamEvaluator]
    hard_stop: Optional[Duration]
    load_path: Optional[str]
    learning_rate: Optional[float]
    global_batch_size: Optional[int]
    rank_microbatch_size: Optional[int]
    warmup_steps: Optional[int]
    profile: bool = False

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
        downstream_evaluators: List[DownstreamEvaluator],
        hard_stop: Optional[Duration] = None,
        load_path: Optional[str] = None,
        global_batch_size: Optional[int] = None,
        rank_microbatch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        metrics_config: Optional[MetricsConfig] = None,
        max_target_sequence_length: int = 8192,
        seed: int = 42,
        warmup_steps: Optional[int] = None,
        s3: bool = True,
        profile: bool = False,
    ):
        self.run_name = run_name
        self.sources = sources
        self.sequence_length = sequence_length
        self.max_tokens = max_tokens
        self.group_id = group_id
        self.seed = seed
        self.model_identifier = model_identifier
        self.tokenizer = self.get_tokenizer_config(tokenizer=tokenizer)
        self.transformer_config = WrappedTransformerConfig.from_model_identifier(model_identifier, self.tokenizer)
        self.beaker_user = beaker_user.strip()
        self.profile = profile
        self.s3 = s3
        self.data_dir: str = "s3://ai2-llm"
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
        self.checkpoint_dir = f"{self.data_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"
        self.eval_interval = eval_interval

        if any(substring in cluster for substring in ["jupiter", "saturn"]) and weka:
            self.root_dir = f"/weka/oe-training-default/ai2-llm"
            logger.info(f"Using Weka bucket as root dir: {self.root_dir}")
            self.checkpoint_dir = f"{self.root_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"

        self.dataset_cache = f"{self.root_dir}/{self.beaker_user.lower()}/{self.run_name}/dataset-cache"

    def get_tokenizer_config(self, tokenizer) -> TokenizerConfig:
        try:
            return Tokenizers[tokenizer].value
        except ValueError as e:
            logger.info(f"Invalid tokenizer identifier: {tokenizer}")
            raise e

    def get_warmup_steps(self) -> int:
        return self.warmup_steps or 2000

    def get_global_batch_size(self) -> int:
        # TODO(undfined): Revisit this logic as it would be nice for this to be automated
        # assert self.sequence_length in {2048, 4096, 8192}
        # seq_len_divisor = self.sequence_length // 2048

        # global_batch_size = 160 * (parameters / 108000000) ** (2 / 3)
        # global_batch_size /= seq_len_divisor
        # global_batch_size /= self.max_dp_world_size
        # global_batch_size = round(global_batch_size)
        # global_batch_size *= self.max_dp_world_size

        # global_batch_size = self.next_power_of_2(self.sequence_length * global_batch_size)

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
        else:
            rbz = 16 * self.sequence_length

        logger.info(f"Rank microbatch size (in tokens) is: {rbz}")

        return rbz

    def get_learning_rate(self) -> float:
        if self.learning_rate is not None:
            lr = self.learning_rate
        else:
            lr = DEFAULT_LR_MAP.get(self.model_identifier.value, 5e-4)

        logger.info(f"Learning rate is: {lr}")

        return lr

    def next_power_of_2(self, x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def build_callbacks(self) -> Dict[str, Callback]:
        callbacks = {
            "gpu_monitor": GPUMemoryMonitorCallback(),
            "garbage_collector": GarbageCollectorCallback(),
            "config_saver": ConfigSaverCallback(),
            "profiler": ProfilerCallback(enabled=self.profile),
            "checkpointer": CheckpointerCallback(
                save_interval=self.save_interval,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        }

        if self.metrics_config:
            if MetricBackend.wandb in self.metrics_config.backends:
                callbacks[MetricBackend.wandb.value] = WandBCallback(
                    name=self.run_name.strip(),
                    project=self.metrics_config.project.strip(),
                    group=self.group_id.strip(),
                    cancel_check_interval=10,
                    enabled=True,
                )
            if MetricBackend.comet in self.metrics_config.backends:
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
                    work_dir=self.dataset_cache,
                ),
                eval_interval=self.eval_interval,
            )

        if self.downstream_evaluators:
            if self.downstream_evaluators[0] == DownstreamEvaluator.ALL:
                evaluators = DownstreamEvaluatorCallbackConfig(
                    tasks=[
                        evaluator.value
                        for evaluator in DownstreamEvaluator
                        if evaluator != DownstreamEvaluator.ALL
                    ],
                    tokenizer=self.tokenizer,
                    eval_interval=self.eval_interval,
                )
            else:
                evaluators = DownstreamEvaluatorCallbackConfig(
                    tasks=[evaluator.value for evaluator in self.downstream_evaluators],
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
            work_dir=self.dataset_cache,
        )

        return dataset_config

    def get_scheduler_config(self, scheduler_type: SchedulerType = SchedulerType.COS_LINEAR) -> Scheduler:
        scheduler_map = {
            SchedulerType.COSINE: lambda: CosWithWarmup(warmup_steps=self.get_warmup_steps()),
            SchedulerType.COS_LINEAR: lambda: CosWithWarmupAndLinearDecay(
                warmup_steps=self.get_warmup_steps(),
            ),
            SchedulerType.WSD: lambda: WSD(
                warmup_steps=self.get_warmup_steps(),
            ),
        }

        return scheduler_map[scheduler_type]()

    def get_optimizer_config(self) -> OptimConfig:
        # TODO(undfined): Add support for other optimizers and allow user to specify optimizer type and properties
        return SkipStepAdamWConfig(
            lr=self.get_learning_rate(),
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))],
        )

    def build(self) -> ModelTrainConfig:
        global_batch_size = self.get_global_batch_size()
        rank_microbatch_size = self.get_rank_microbatch_size()
        dataset_config = self.build_dataset_config()
        optim_config = self.get_optimizer_config()
        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=global_batch_size,
            work_dir=self.dataset_cache,
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
            work_dir=self.dataset_cache,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=5,
            max_duration=Duration.tokens(self.max_tokens),
        )

        for callback_name, callback in self.build_callbacks().items():
            trainer_config.callbacks[callback_name] = callback

        return ModelTrainConfig(
            init_seed=self.seed,
            model=self.transformer_config,
            optim=optim_config,
            dataset=dataset_config,
            data_loader=data_loader_config,
            trainer=trainer_config,
            train_module=train_module_config,
        )
