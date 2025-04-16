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


class Tokenizers(Enum):
    dolma2 = TokenizerConfig.dolma2()
    gpt_neox = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    superbpe_experimental = TokenizerConfig(
        vocab_size=180021,
        identifier="superbpe-experimental_v0.1.0",
        bos_token_id=180000,
        eos_token_id=180000,
        pad_token_id=180001,
    )


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


class ModelConfigIdentifier:
    """
    A dynamic registry for model identifiers that auto-initializes when used.
    """

    _registry: dict[str, str] = {}
    _initialized = False

    def __init__(self, identifier):
        # Auto-initialize the first time this class is used
        if not ModelConfigIdentifier._initialized:
            ModelConfigIdentifier._initialize_identifiers()

        if identifier not in ModelConfigIdentifier._registry:
            raise ValueError(
                f"'{identifier}' is not a valid model identifier. "
                f"Available models: {', '.join(ModelConfigIdentifier._registry.keys())}"
            )

        self.value = identifier
        self.name = identifier

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"ModelConfigIdentifier({self.value!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, ModelConfigIdentifier):
            return self.value == other.value
        return False

    @classmethod
    def _get_model_methods(cls, target_class) -> list[str]:
        """Get all classmethods of a class that might represent model configurations."""
        return [
            attr
            for attr in dir(target_class)
            if callable(getattr(target_class, attr))
            and not attr.startswith("_")
            and attr not in ["from_dict", "from_json", "from_model_identifier", "values", "keys"]
        ]

    @classmethod
    def _initialize_identifiers(cls) -> None:
        """Initialize the model identifier registry with methods from TransformerConfig and WrappedTransformerConfig."""
        # Add default models
        cls._registry["default"] = "default"

        # Add methods from WrappedTransformerConfig
        for method_name in cls._get_model_methods(WrappedTransformerConfig):
            cls._registry[method_name] = method_name

        # Add methods from TransformerConfig
        for method_name in cls._get_model_methods(TransformerConfig):
            if method_name not in cls._registry:
                cls._registry[method_name] = method_name

        cls._initialized = True

    @classmethod
    def keys(cls) -> list[str]:
        """Return all valid model identifier keys."""
        if not cls._initialized:
            cls._initialize_identifiers()
        return list(cls._registry.keys())

    @classmethod
    def values(cls) -> list[str]:
        """Return all valid model identifier values."""
        if not cls._initialized:
            cls._initialize_identifiers()
        return list(cls._registry.values())

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema

        def validate_identifier(value, info):
            # Ensure registry is initialized
            if not cls._initialized:
                cls._initialize_identifiers()

            # Handle existing instances
            if isinstance(value, cls):
                return value

            # Handle string values
            if not isinstance(value, str):
                raise ValueError(f"Expected string or {cls.__name__}, got {type(value)}")

            # Validate against registry
            if value not in cls._registry:
                valid_values = ", ".join(cls._registry.keys())
                raise ValueError(
                    f"'{value}' is not a valid model identifier. " f"Available models: {valid_values}"
                )

            return cls(value)

        return core_schema.with_info_plain_validator_function(
            validate_identifier,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda instance: instance.value),
            metadata={
                "type": "enum-like",
                "values": list(cls.keys()),
            },
        )


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


DEFAULT_LR_MAP = {
    "olmo2_1B": 1.8e-3,
}
