from dataclasses import dataclass
from enum import Enum

import olmo_core.train.train_module as train_module
from olmo_core.config import Config
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig, TokenizerConfig
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.rope import ABFRoPEScalingConfig, YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerBlockType, TransformerConfig
from olmo_core.optim import OptimConfig
from olmo_core.train import TrainerConfig


class Tokenizers(Enum):
    dolma2 = TokenizerConfig.dolma2()
    gpt_neox = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    superbpe_experimental = TokenizerConfig(
        vocab_size=180021,
        identifier="allenai/superbpe-experimental_v0.1.0",
        eos_token_id=180000,
        pad_token_id=180001,
    )
    dolma2_180k = TokenizerConfig(
        vocab_size=180021,
        identifier="allenai/dolma2-180k-experimental-0.0.1",
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
                raise ValueError(f"'{value}' is not a valid model identifier. Available models: {valid_values}")

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
    def olmo25_7b(cls, tokenizer: TokenizerConfig) -> TransformerConfig:
        """
        OLMo2.5 retrofit
        """
        config = TransformerConfig.olmo2_7B(vocab_size=tokenizer.padded_vocab_size())
        config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[4096, 4096, 4096, -1],
        )
        config.block.attention.use_flash = True
        return config

    @classmethod
    def olmo25_7b_fullattn(cls, tokenizer: TokenizerConfig) -> TransformerConfig:
        """
        OLMo2.5 retrofit w/ full attn 
        """
        config = TransformerConfig.olmo2_7B(vocab_size=tokenizer.padded_vocab_size())
        config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[-1, -1, -1, -1],
        )
        config.block.attention.use_flash = True
        return config

    @classmethod
    def olmo3_7B_swafix(cls, tokenizer: TokenizerConfig) -> TransformerConfig:
        """
        Temporary OLMo3 7B "swafix" config until it is merged into olmo-core
        https://github.com/allenai/OLMo-core/pull/310/files#diff-03f6a1f5db18fc4be7a243d8168698ae674cd50b2866253bcdadba5d48590b3dR48
        """
        config = getattr(TransformerConfig, "olmo2_7B")(
            n_kv_heads=8,
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=1024,
            vocab_size=tokenizer.padded_vocab_size(),
        )
        config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[4096, 4096, 4096, -1],
        )
        config.block.attention.use_flash = True
        config.block.attention.use_head_qk_norm = True
        return config

    @classmethod
    def olmo3_7B_swafix_yolofull(cls, tokenizer: TokenizerConfig) -> TransformerConfig:
        """
        Temporary OLMo3 7B "swafix" config with full attn until it is merged into olmo-core
        https://github.com/allenai/OLMo-core/pull/310/files#diff-03f6a1f5db18fc4be7a243d8168698ae674cd50b2866253bcdadba5d48590b3dR48
        """
        config = getattr(TransformerConfig, "olmo2_7B")(
            n_kv_heads=8,
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=1024,
            vocab_size=tokenizer.padded_vocab_size(),
        )
        config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[-1, -1, -1, -1],
        )
        config.block.attention.use_flash = True
        config.block.attention.use_head_qk_norm = True
        return config

    @classmethod
    def olmo3_7B_swafix_abf(cls, tokenizer: TokenizerConfig) -> TransformerConfig:
        """
        Temporary OLMo3 7B "swafix" config until it is merged into olmo-core
        https://github.com/allenai/OLMo-core/pull/310/files#diff-03f6a1f5db18fc4be7a243d8168698ae674cd50b2866253bcdadba5d48590b3dR48
        """
        config = getattr(TransformerConfig, "olmo2_7B")(
            n_kv_heads=8,
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=1024,
            vocab_size=tokenizer.padded_vocab_size(),
            rope_scaling=ABFRoPEScalingConfig(new_theta=8000000),
        )
        config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[4096, 4096, 4096, -1],
        )
        config.block.attention.use_flash = True
        config.block.attention.use_head_qk_norm = True
        return config

    @classmethod
    def olmo3_7B_swafix_abf_fullonly(cls, tokenizer: TokenizerConfig) -> TransformerConfig:
        """
        Temporary OLMo3 7B "swafix" config until it is merged into olmo-core
        https://github.com/allenai/OLMo-core/pull/310/files#diff-03f6a1f5db18fc4be7a243d8168698ae674cd50b2866253bcdadba5d48590b3dR48
        """
        config = getattr(TransformerConfig, "olmo2_7B")(
            n_kv_heads=8,
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=1024,
            vocab_size=tokenizer.padded_vocab_size(),
            rope_scaling=ABFRoPEScalingConfig(new_theta=8000000),
        )
        config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[4096, 4096, 4096, -1],
        )
        config.block.attention.use_flash = True
        config.block.attention.use_head_qk_norm = True

        def no_rope_scaling(block: TransformerBlockConfig) -> TransformerBlockConfig:
            rope_config = block.attention.rope
            if rope_config is not None:
                rope_config.scaling = None
                block.attention.rope = rope_config
            return block

        config.block_overrides = {
            i: no_rope_scaling(config.block.copy())
            for i in range(config.n_layers)
            if config.block.attention.sliding_window.should_use_swa(i, config.n_layers)
        }
        return config

    @classmethod
    def olmo3_7B_swafix_yarn(cls, tokenizer: TokenizerConfig) -> TransformerConfig:
        """
        Temporary OLMo3 7B "swafix" config until it is merged into olmo-core
        https://github.com/allenai/OLMo-core/pull/310/files#diff-03f6a1f5db18fc4be7a243d8168698ae674cd50b2866253bcdadba5d48590b3dR48
        """
        config = getattr(TransformerConfig, "olmo2_7B")(
            n_kv_heads=8,
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=1024,
            vocab_size=tokenizer.padded_vocab_size(),
            rope_scaling=YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192),
        )
        config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[4096, 4096, 4096, -1],
        )
        config.block.attention.use_flash = True
        config.block.attention.use_head_qk_norm = True
        return config

    @classmethod
    def olmo3_7B_swafix_yarn_fullonly(cls, tokenizer: TokenizerConfig) -> TransformerConfig:
        """
        Temporary OLMo3 7B "swafix" config until it is merged into olmo-core
        https://github.com/allenai/OLMo-core/pull/310/files#diff-03f6a1f5db18fc4be7a243d8168698ae674cd50b2866253bcdadba5d48590b3dR48
        """
        config = getattr(TransformerConfig, "olmo2_7B")(
            n_kv_heads=8,
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=1024,
            vocab_size=tokenizer.padded_vocab_size(),
            rope_scaling=YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192),
        )
        config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[4096, 4096, 4096, -1],
        )
        config.block.attention.use_flash = True
        config.block.attention.use_head_qk_norm = True

        def no_rope_scaling(block: TransformerBlockConfig) -> TransformerBlockConfig:
            rope_config = block.attention.rope
            if rope_config is not None:
                rope_config.scaling = None
                block.attention.rope = rope_config
            return block

        config.block_overrides = {
            i: no_rope_scaling(config.block.copy())
            for i in range(config.n_layers)
            if config.block.attention.sliding_window.should_use_swa(i, config.n_layers)
        }
        return config

    @classmethod
    def from_model_identifier(
        cls,
        model_identifier: ModelConfigIdentifier,
        tokenizer: TokenizerConfig = Tokenizers.dolma2.value,
    ) -> TransformerConfig:
        """
        Create a TransformerConfig from a ModelConfigIdentifier.

        This method supports all models defined in the ModelConfigIdentifier enum by
        mapping them to appropriate TransformerConfig class methods.

        Args:
            model_identifier: The model identifier to create a config for
            tokenizer: The tokenizer config to use
            model_overrides: Optional overrides for the model config

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
    "olmo2_1B_v2": 1.8e-3,
}
