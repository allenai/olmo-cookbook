from dataclasses import dataclass
from enum import Enum
from typing import Optional

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import (
    TransformerBlockType,
    TransformerConfig,
    TransformerDataParallelConfig,
)
from olmo_core.optim import AdamWConfig
from olmo_core.train import TrainerConfig


@dataclass
class ModelTrainConfig(Config):
    model: TransformerConfig
    optim: AdamWConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536


@dataclass
class DefaultOptimizerProperties:
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1


@dataclass
class DefaultTransformerProperties:
    block_type: TransformerBlockType = TransformerBlockType.reordered_norm
    compile: bool = True
    decay_embeddings: bool = False
    dp_type: DataParallelType = DataParallelType.fsdp
    layer_norm_eps: float = 1e-6
    qk_norm: bool = True
    rope_theta: int = 500_000


class WrappedTransformerConfig(TransformerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def olmo_30m(cls, tokenizer: TokenizerConfig) -> "WrappedTransformerConfig":
        """
        OLMo 30m
        """
        return WrappedTransformerConfig(
            **getattr(TransformerConfig, "llama_like")(
                d_model=256,
                n_heads=8,
                n_layers=4,
                vocab_size=tokenizer.padded_vocab_size(),
                compile=DefaultTransformerProperties.compile,
                rope_theta=DefaultTransformerProperties.rope_theta,
                layer_norm_eps=DefaultTransformerProperties.layer_norm_eps,
                qk_norm=DefaultTransformerProperties.qk_norm,
                block_name=DefaultTransformerProperties.block_type,
                dp_config=TransformerDataParallelConfig(
                    name=DefaultTransformerProperties.dp_type,
                    param_dtype=DType.bfloat16,
                    reduce_dtype=DType.float32,
                ),
            )
        )

    @classmethod
    def olmo2_core_190M(cls, dp_type: Optional[DataParallelType] = None) -> "WrappedTransformerConfig":
        return WrappedTransformerConfig(
            **getattr(TransformerConfig, "olmo2_190M")(
                compile=True,
                dp_config=TransformerDataParallelConfig(
                    name=dp_type if dp_type else DefaultTransformerProperties.dp_type,
                    param_dtype=DType.bfloat16,
                    reduce_dtype=DType.float32,
                ),
            ),
        )

    @classmethod
    def olmo2_core_1B(cls, dp_type: Optional[DataParallelType] = None) -> "WrappedTransformerConfig":
        """
        OLMo2 1b (1_336_035_328 parameters)
        """
        return WrappedTransformerConfig(
            **getattr(TransformerConfig, "olmo2_1B")(
                compile=True,
                dp_config=TransformerDataParallelConfig(
                    name=dp_type if dp_type else DefaultTransformerProperties.dp_type,
                    param_dtype=DType.bfloat16,
                    reduce_dtype=DType.float32,
                ),
            ),
        )

    @classmethod
    def from_model_identifier(cls, model_identifier: str) -> "WrappedTransformerConfig":
        if model_identifier == "olmo_30m":
            return cls.olmo_30m(TokenizerConfig.gpt_neox_olmo_dolma_v1_5())
        elif model_identifier == "olmo2_190M":
            return cls.olmo2_core_190M()
        elif model_identifier == "olmo2_1B":
            return cls.olmo2_core_1B()
        else:
            raise ValueError(f"Model identifier {model_identifier} is not supported.")


class SupportedTokenizers(Enum):
    dolma2 = TokenizerConfig.dolma2()
    gpt_neox = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
