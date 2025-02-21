import logging
import os
from dataclasses import dataclass
from typing import Dict, List

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SchedulerCallback,
    WandBCallback,
)

from cookbook.aliases import SourceInstance
from cookbook.data.dataset import MixtureBuilder
from cookbook.model.aliases import (
    ModelConfig,
    ModelTrainConfig,
    SupportedModels,
    SupportedTokenizers,
)
from cookbook.model.evaluators import DownstreamEvaluators

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfigBuilder:
    """
    A builder class for configuring and creating a transformer model training configuration.

    Attributes:
        run_name (str): The name of the run.
        sources (List[SourceInstance]): A list of source instances.
        sequence_length (int): The sequence length for the model.
        max_tokens (int): The maximum number of tokens to be processed in a batch.
        model_config (ModelConfig): The model configuration.
        group_id (str): The group ID for the run.
        cluster (str): The cluster name.
        beaker_user (str): The Beaker user name.
        s3 (bool): Whether to use S3 for storage.
        seed (int): The random seed for reproducibility. Default is 42.
        tokenizer (TokenizerConfig): The tokenizer configuration.
        dtype (str): The data type for the dataset.
        profile (bool): Whether to enable profiling. Default is False.
        weka (bool): Whether to use Weka buckets. Default is False.

    Methods:
        __init__(run_name, sources, sequence_length, max_tokens, group_id, cluster, beaker_user,
                 tokenizer, dtype, model_identifier, seed=42, s3=True, profile=False):
            Initializes the TransformerConfigBuilder.

        get_tokenizer_config(tokenizer: str) -> TokenizerConfig:
            Returns the tokenizer configuration based on the tokenizer identifier.

        get_warmup_steps(parameters: int) -> int:
            Returns the number of warmup steps based on the model parameters.

        get_batch_size(parameters: int) -> int:
            Returns the global batch size based on the sequence length and model parameters.

        next_power_of_2(x: int) -> int:
            Returns the next power of 2 greater than or equal to x.

        build_callbacks(model: TransformerConfig) -> Dict[str, Callback]:
            Builds and returns a dictionary of callbacks for the trainer.

        build() -> ModelTrainConfig:
            Builds and returns the model training configuration.
    """

    run_name: str
    sources: List[SourceInstance]
    sequence_length: int
    max_tokens: int
    model_config: ModelConfig
    group_id: str
    cluster: str
    beaker_user: str
    s3: bool
    seed: int
    tokenizer: TokenizerConfig
    dtype: str
    weka: bool
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
        self.model_config = SupportedModels[model_identifier].value
        self.beaker_user = beaker_user.strip()
        self.profile = profile
        self.s3 = s3
        self.tokenizer = self.get_tokenizer_config(tokenizer=tokenizer)
        self.data_dir: str = "s3://ai2-llm"
        self.dataset_dtype = NumpyDatasetDType[dtype]
        self.root_dir = f"/tmp/{self.run_name}"

        self.checkpoint_dir = (
            f"{self.data_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"
        )

        if any(substring in cluster for substring in ["jupiter", "saturn"]) and weka:
            logger.info("Using Weka bucket as root dir")
            self.root_dir = f"/weka/oe-training-default/ai2-llm"

        self.dataset_cache = (
            f"{self.root_dir}/{self.beaker_user.lower()}/{self.run_name}/dataset-cache"
        )

    def get_tokenizer_config(self, tokenizer) -> TokenizerConfig:
        try:
            return SupportedTokenizers[tokenizer].value
        except ValueError as e:
            logger.info(f"Invalid tokenizer identifier: {tokenizer}")
            raise e

    def get_warmup_steps(self, parameters: int) -> int:
        return round(
            parameters / (self.get_batch_size(parameters) * self.model_config.max_sequence_length)
        )

    def get_batch_size(self, parameters: int) -> int:
        if self.sequence_length != 2048:
            raise NotImplementedError("Only sequence length 2048 is supported right now")

        global_batch_size = 160 * (parameters / 108000000) ** (2 / 3)
        global_batch_size /= self.model_config.batch_divisor
        global_batch_size = round(global_batch_size)
        global_batch_size *= self.model_config.batch_divisor

        global_batch_size = self.next_power_of_2(global_batch_size)
        print(f"Global batch size is: {global_batch_size}")

        return global_batch_size

    def next_power_of_2(self, x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def build_callbacks(self, model: TransformerConfig) -> Dict[str, Callback]:
        return {
            "lr_scheduler": SchedulerCallback(
                scheduler=CosWithWarmup(
                    warmup_steps=self.get_warmup_steps(model.num_params), t_max=self.max_tokens
                ),
            ),
            "gpu_monitor": GPUMemoryMonitorCallback(),
            "grad_clipper": GradClipperCallback(max_grad_norm=self.model_config.max_grad_norm),
            "config_saver": ConfigSaverCallback(),
            "profiler": ProfilerCallback(enabled=self.profile),
            "checkpointer": CheckpointerCallback(
                save_interval=self.model_config.save_interval,
                ephemeral_save_interval=100,
                save_async=True,
            ),
            "wandb": WandBCallback(
                name=self.run_name.strip(),
                project="olmo-cookbook",
                group=self.group_id.strip(),
                cancel_check_interval=10,
                enabled=True,
            ),
            # TODO: Add Comet ML callback
            "lm_evaluator": LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    name=NumpyDatasetType.padded_fsl,
                    mix_base_dir=self.data_dir,
                    sequence_length=self.sequence_length,
                    tokenizer=self.tokenizer,
                    work_dir=self.dataset_cache,
                ),
                eval_interval=self.model_config.eval_interval,
            ),
            # TODO: Make this configurable?
            "downstream_evaluator": DownstreamEvaluatorCallbackConfig(
                tasks=[task.value for task in DownstreamEvaluators],
                tokenizer=self.tokenizer,
                eval_interval=self.model_config.eval_interval,
            ),
        }

    def build(self) -> ModelTrainConfig:
        tokenizer = self.tokenizer
        model = TransformerConfig.llama_like(
            compile=self.model_config.compile,
            d_model=self.model_config.d_model,
            n_layers=self.model_config.n_layers,
            n_heads=self.model_config.n_heads,
            vocab_size=tokenizer.padded_vocab_size(),
            rope_theta=self.model_config.rope_theta,
            layer_norm_eps=self.model_config.layer_norm_eps,
            qk_norm=self.model_config.qk_norm,
            block_name=self.model_config.block_type,
            dp_config=TransformerDataParallelConfig(
                name=self.model_config.dp_type,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
            ),
        )

        global_batch_size = self.get_batch_size(model.num_params)
        learning_rate = 4.7e-3 * (model.num_params / tokenizer.padded_vocab_size()) ** (-1 / 3)

        if self.sequence_length == 4096:
            learning_rate /= 4
            raise NotImplementedError("Only sequence length 2048 is supported right now")

        optim_config = AdamWConfig(
            lr=learning_rate,
            eps=self.model_config.eps,
            betas=self.model_config.betas,
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"],
                    opts=dict(weight_decay=self.model_config.weight_decay),
                )
            ],
        )

        # mixture_config = MixtureBuilder(
        #     sources=self.sources,
        #     max_tokens=self.max_tokens,
        #     sequence_length=self.sequence_length,
        #     seed=self.seed,
        #     processes=min(os.cpu_count() or 1, 16),
        #     dtype=self.dataset_dtype,
        # ).build()

        paths = []
        for source in self.sources:
            paths.extend(source.paths)

        dataset_config = NumpyDatasetConfig(
            paths=paths,
            name=NumpyDatasetType.fsl,
            sequence_length=self.sequence_length,
            tokenizer=tokenizer,
            work_dir=self.dataset_cache,
        )

        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=global_batch_size * self.sequence_length,
            work_dir=self.dataset_cache,
            seed=self.seed,
            num_workers=16,
        )

        trainer_config = TrainerConfig(
            save_folder=self.checkpoint_dir,
            work_dir=self.dataset_cache,
            rank_microbatch_size=self.model_config.device_batch_size * self.sequence_length,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=5,
        )

        for callback_name, callback in self.build_callbacks(model).items():
            trainer_config.callbacks[callback_name] = callback

        return ModelTrainConfig(
            model=model,
            optim=optim_config,
            dataset=dataset_config,
            data_loader=data_loader_config,
            trainer=trainer_config,
        )
