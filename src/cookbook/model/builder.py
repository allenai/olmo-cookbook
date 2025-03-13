import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride, Scheduler
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SchedulerCallback,
    WandBCallback,
)
from olmo_core.train.common import LoadStrategy

from cookbook.aliases import SourceInstance, WandbConfig
from cookbook.data.dataset import MixtureBuilder
from cookbook.model.config import (
    MODEL_TO_LR_MAP,
    DefaultOptimizerProperties,
    ModelTrainConfig,
    SupportedTokenizers,
    WrappedTransformerConfig,
)
from cookbook.model.evaluators import DownstreamEvaluators
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
        model_identifier (str): The identifier for the model.
        transformer_config (TransformerConfig): The transformer configuration.
        group_id (str): The group ID for the run.
        cluster (str): The cluster name.
        beaker_user (str): The Beaker user name.
        s3 (bool): Whether to use S3 for storage.
        seed (int): The random seed for reproducibility.
        tokenizer (TokenizerConfig): The tokenizer configuration.
        dtype (str): The data type for the dataset.
        weka (bool): Whether to use Weka buckets.
        wandb_config (Optional[WandbConfig]): The Weights and Biases configuration.
        max_dp_world_size (int): The maximum data parallel world size.
        eval_interval (int): The evaluation interval.
        save_interval (int): The save interval.
        lm_evaluator (bool): Whether to enable language model evaluation.
        downstream_evaluators (List[DownstreamEvaluators]): The downstream evaluators.
        learning_rate (Optional[float]): The learning rate for the optimizer.
        profile (bool): Whether to enable profiling.

    Methods:
        __init__(run_name, sources, sequence_length, max_tokens, group_id, cluster, beaker_user,
                 tokenizer, dtype, model_identifier, weka, max_dp_world_size, save_interval, eval_interval,
                 lm_evaluator, downstream_evaluators, learning_rate=None, wandb_config=None,
                 max_target_sequence_length=8192, seed=42, s3=True, profile=False):
            Initializes the TransformerConfigBuilder.

        get_tokenizer_config(tokenizer: str) -> TokenizerConfig:
            Returns the tokenizer configuration based on the tokenizer identifier.

        get_warmup_steps() -> int:
            Returns the number of warmup steps.

        get_batch_size() -> int:
            Returns the global batch size based on the sequence length.

        next_power_of_2(x: int) -> int:
            Returns the next power of 2 greater than or equal to x.

        get_learning_rate() -> float:
            Returns the learning rate for the optimizer.

        build_callbacks(model: TransformerConfig) -> Dict[str, Callback]:
            Builds and returns a dictionary of callbacks for the trainer.

        build_dataset_config() -> NumpyDatasetConfig:
            Builds and returns the dataset configuration.

        get_scheduler_config(scheduler_type: str = "cosine") -> Scheduler:
            Returns the scheduler configuration based on the scheduler type.

        get_optimizer_config() -> AdamWConfig:
            Returns the optimizer configuration.

        build() -> ModelTrainConfig:
            Builds and returns the model training configuration.
    """

    run_name: str
    sources: List[SourceInstance]
    sequence_length: int
    max_target_sequence_length: int
    max_tokens: int
    model_identifier: str
    transformer_config: TransformerConfig
    group_id: str
    cluster: str
    beaker_user: str
    s3: bool
    seed: int
    tokenizer: TokenizerConfig
    dtype: str
    weka: bool
    wandb_config: Optional[WandbConfig]
    max_dp_world_size: int
    eval_interval: int
    save_interval: int
    lm_evaluator: bool
    downstream_evaluators: List[DownstreamEvaluators]
    load_path: Optional[str]
    learning_rate: Optional[float]
    global_batch_size: Optional[int]
    rank_microbatch_size: Optional[int]
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
        model_identifier: str,
        weka: bool,
        max_dp_world_size: int,
        save_interval: int,
        eval_interval: int,
        lm_evaluator: bool,
        downstream_evaluators: List[DownstreamEvaluators],
        load_path: Optional[str] = None,
        global_batch_size: Optional[int] = None,
        rank_microbatch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        wandb_config: Optional[WandbConfig] = None,
        max_target_sequence_length: int = 8192,
        seed: int = 42,
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
        self.transformer_config = WrappedTransformerConfig.from_model_identifier(model_identifier)
        self.beaker_user = beaker_user.strip()
        self.profile = profile
        self.s3 = s3
        self.tokenizer = self.get_tokenizer_config(tokenizer=tokenizer)
        self.data_dir: str = "s3://ai2-llm"
        self.dataset_dtype = NumpyDatasetDType[dtype]
        self.root_dir = f"/tmp/{self.run_name}"
        self.wandb_config = wandb_config
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
        self.load_path = load_path
        self.checkpoint_dir = f"{self.data_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"
        self.eval_interval = eval_interval

        if any(substring in cluster for substring in ["jupiter", "saturn"]) and weka:
            self.root_dir = f"/weka/oe-training-default/ai2-llm"
            logger.info(f"Using Weka bucket as root dir: {self.root_dir}")
            self.checkpoint_dir = f"{self.root_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"

        self.dataset_cache = f"{self.root_dir}/{self.beaker_user.lower()}/{self.run_name}/dataset-cache"

    def get_tokenizer_config(self, tokenizer) -> TokenizerConfig:
        try:
            return SupportedTokenizers[tokenizer].value
        except ValueError as e:
            logger.info(f"Invalid tokenizer identifier: {tokenizer}")
            raise e

    def get_warmup_steps(self) -> int:
        return 2000

    def get_batch_sizes(self) -> tuple[int, int]:
        # assert self.sequence_length in {2048, 4096, 8192}
        # seq_len_divisor = self.sequence_length // 2048

        # global_batch_size = 160 * (parameters / 108000000) ** (2 / 3)
        # global_batch_size /= seq_len_divisor
        # global_batch_size /= self.max_dp_world_size
        # global_batch_size = round(global_batch_size)
        # global_batch_size *= self.max_dp_world_size

        # global_batch_size = self.next_power_of_2(self.sequence_length * global_batch_size)
        if self.global_batch_size:
            global_batch_size = self.global_batch_size * self.global_batch_size
        else:
            global_batch_size = 1024 * self.sequence_length

        if self.rank_microbatch_size:
            rank_microbatch_size = self.rank_microbatch_size * self.rank_microbatch_size
        else:
            rank_microbatch_size = 16 * self.sequence_length

        print(f"Global batch size (in tokens) is: {global_batch_size}")

        return global_batch_size, rank_microbatch_size

    # def default_learning_rate(self) -> float:
    #     learning_rate = 4.7e-3 * (self.transformer_config.num_params / self.tokenizer.padded_vocab_size()) ** (
    #         -1 / 3
    #     )
    #     learning_rate = learning_rate / self.sequence_length
    #     return learning_rate

    def get_learning_rate(self) -> float:
        if self.learning_rate:
            return self.learning_rate

        maybe_lr = MODEL_TO_LR_MAP.get(self.model_identifier)

        if maybe_lr:
            return maybe_lr
        else:
            return 5e-4

    def next_power_of_2(self, x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def build_callbacks(self, model: TransformerConfig) -> Dict[str, Callback]:
        callbacks = {
            "lr_scheduler": SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=self.get_warmup_steps())),
            "gpu_monitor": GPUMemoryMonitorCallback(),
            "grad_clipper": GradClipperCallback(max_grad_norm=self.max_grad_norm),
            "garbage_collector": GarbageCollectorCallback(),
            "config_saver": ConfigSaverCallback(),
            "profiler": ProfilerCallback(enabled=self.profile),
            "checkpointer": CheckpointerCallback(
                save_interval=self.save_interval,
                ephemeral_save_interval=100,
                save_async=True,
            ),
            "wandb": WandBCallback(
                name=self.run_name.strip(),
                project=self.wandb_config.project.strip() if self.wandb_config else "olmo-cookbook",
                group=self.group_id.strip(),
                cancel_check_interval=10,
                enabled=True,
            ),
            # TODO(undfined): Add Comet ML callback
        }

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
            callbacks["downstream_evaluator"] = DownstreamEvaluatorCallbackConfig(
                tasks=[evaluator.value for evaluator in self.downstream_evaluators],
                tokenizer=self.tokenizer,
                eval_interval=self.eval_interval,
            )

        return callbacks

    def build_dataset_config(self) -> NumpyDatasetConfig:
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
                processes=12,
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

    def get_scheduler_config(self, scheduler_type: str = "cosine") -> Scheduler:
        if scheduler_type == "cosine":
            return CosWithWarmup(warmup_steps=self.get_warmup_steps())
        elif scheduler_type == "wsd":
            return WSD(
                warmup_steps=self.get_warmup_steps(),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def get_optimizer_config(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.get_learning_rate(),
            eps=DefaultOptimizerProperties.eps,
            betas=DefaultOptimizerProperties.betas,
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"],
                    opts=dict(weight_decay=0.0),
                )
            ],
            fused=True,
            weight_decay=DefaultOptimizerProperties.weight_decay,
        )

    def build(self) -> ModelTrainConfig:
        global_batch_size, rank_microbatch_size = self.get_batch_sizes()
        learning_rate = self.get_learning_rate()

        if self.sequence_length == 4096:
            learning_rate /= 4

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

        trainer_config = TrainerConfig(
            load_path=load_path,
            load_strategy=load_strategy,
            save_folder=self.checkpoint_dir,
            work_dir=self.dataset_cache,
            rank_microbatch_size=rank_microbatch_size,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=5,
            compile_loss=True,
            z_loss_multiplier=1e-5,
            max_duration=Duration.tokens(self.max_tokens),
        )

        for callback_name, callback in self.build_callbacks(self.transformer_config).items():
            trainer_config.callbacks[callback_name] = callback

        return ModelTrainConfig(
            init_seed=self.seed,
            model=self.transformer_config,
            optim=optim_config,
            dataset=dataset_config,
            data_loader=data_loader_config,
            trainer=trainer_config,
        )
