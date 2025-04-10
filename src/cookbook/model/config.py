from dataclasses import dataclass
from enum import Enum

import olmo_core.train.train_module as train_module
from olmo_core.config import Config
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig, TokenizerConfig
from olmo_core.nn.transformer import (
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.optim import OptimConfig
from olmo_core.train import TrainerConfig


@dataclass
class ModelTrainConfig(Config):
    model: TransformerConfig
    optim: OptimConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: train_module.TransformerTrainModuleConfig
    init_seed: int = 12536


@dataclass
class DefaultOptimizerProperties:
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1


@dataclass
class DefaultTransformerProperties:
    block_type: TransformerBlockType = TransformerBlockType.reordered_norm
    decay_embeddings: bool = False
    layer_norm_eps: float = 1e-6
    qk_norm: bool = True
    rope_theta: int = 500_000


class WrappedTransformerConfig:
    @classmethod
    def olmo_30m(cls, tokenizer: TokenizerConfig) -> TransformerConfig:
        """
        OLMo 30m
        """
        return getattr(TransformerConfig, "llama_like")(
            d_model=256,
            n_heads=8,
            n_layers=4,
            vocab_size=tokenizer.padded_vocab_size(),
            rope_theta=DefaultTransformerProperties.rope_theta,
            layer_norm_eps=DefaultTransformerProperties.layer_norm_eps,
            qk_norm=DefaultTransformerProperties.qk_norm,
            block_name=DefaultTransformerProperties.block_type,
        )

    @classmethod
    def olmo2_core_190M(cls) -> TransformerConfig:
        return getattr(TransformerConfig, "olmo2_190M")(
            vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
        )

    @classmethod
    def olmo2_core_1B(cls) -> TransformerConfig:
        """
        OLMo2 1b (1_336_035_328 parameters)
        """
        return getattr(TransformerConfig, "olmo2_1B")(
            vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
        )

    @classmethod
    def from_model_identifier(cls, model_identifier: str) -> TransformerConfig:
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


MODEL_TO_LR_MAP = {
    "olmo2_1B": 1.8e-3,
}
