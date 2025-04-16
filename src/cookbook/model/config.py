from dataclasses import dataclass
from enum import Enum
from typing import Optional

import olmo_core.train.train_module as train_module
from olmo_core.config import Config
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig, TokenizerConfig
from olmo_core.nn.transformer import (
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.optim import OptimConfig
from olmo_core.train import TrainerConfig


class Tokenizers(Enum):
    dolma2 = TokenizerConfig.dolma2()
    gpt_neox = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    superbpe_experimental = TokenizerConfig.from_hf("allenai/superbpe-experimental_v0.1.0")


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


class ModelConfigIdentifier(Enum):
    @classmethod
    def _get_model_methods(cls, target_class):
        """Get all classmethods of a class that might represent model configurations."""
        return [
            attr
            for attr in dir(target_class)
            if callable(getattr(target_class, attr))
            and not attr.startswith("_")
            and attr not in ["from_dict", "from_json", "from_model_identifier", "values", "keys"]
        ]

    # Dynamically add model names from WrappedTransformerConfig
    @classmethod
    def values(cls):
        return [e.value for e in cls]

    @classmethod
    def keys(cls):
        return [e.name for e in cls]


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
    def from_model_identifier(
        cls, model_identifier: ModelConfigIdentifier, tokenizer: TokenizerConfig = Tokenizers.dolma2.value
    ) -> TransformerConfig:
        """
        Create a TransformerConfig from a ModelConfigIdentifier.

        This method supports all models defined in the ModelConfigIdentifier enum by
        mapping them to appropriate TransformerConfig class methods.

        Args:
            model_identifier: The model identifier to create a config for
            tokenizer: The tokenizer config to use

        Returns:
            A TransformerConfig instance for the specified model

        Raises:
            ValueError: If the model identifier isn't supported in either cookbook or olmo-core
        """
        model_name = model_identifier.value

        # First, check if we have a custom config override for this model
        if hasattr(cls, model_name):
            return getattr(cls, model_name)(tokenizer)

        # Then, check if the TransformerConfig class has a method for this model
        if hasattr(TransformerConfig, model_name):
            return getattr(TransformerConfig, model_name)(
                vocab_size=tokenizer.padded_vocab_size(),
            )

        raise ValueError(
            f"Model identifier '{model_identifier}' is not supported in either cookbook or olmo-core."
            f" Available models: {', '.join(ModelConfigIdentifier.keys())}"
        )


# NOTE: This function initializes the ModelConfigIdentifier enum with methods from
# both WrappedTransformerConfig and TransformerConfig so that we can use any of them as identifiers.
def _initialize_model_config_identifiers():
    for method_name in ModelConfigIdentifier._get_model_methods(WrappedTransformerConfig):
        setattr(ModelConfigIdentifier, method_name, method_name)

    for method_name in ModelConfigIdentifier._get_model_methods(TransformerConfig):
        if not hasattr(ModelConfigIdentifier, method_name):  # Avoid duplicates
            setattr(ModelConfigIdentifier, method_name, method_name)


_initialize_model_config_identifiers()


DEFAULT_LR_MAP = {
    "olmo2_1B": 1.8e-3,
}
