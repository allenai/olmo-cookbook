from dataclasses import dataclass
from enum import Enum

from olmo_core.config import Config
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig
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
class ModelConfig:
    compile: bool
    d_model: int
    n_heads: int
    n_layers: int
    rope_theta: int
    flash_attention: bool
    max_sequence_length: int
    layer_norm_eps: float = 1e-6
    save_interval: int = 1000
    eval_interval: int = 200
    device_batch_size: int = 8
    batch_divisor: int = 32
    eps: float = 1e-8
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    decay_embeddings: bool = False
    qk_norm: bool = True
    dp_type: DataParallelType = DataParallelType.fsdp
    block_type: TransformerBlockType = TransformerBlockType.reordered_norm

    @classmethod
    def olmo_30m(cls) -> "ModelConfig":
        return ModelConfig(
            compile=True,
            d_model=256,
            n_heads=8,
            n_layers=4,
            rope_theta=500_000,
            flash_attention=True,
            max_sequence_length=4096,
        )

    @classmethod
    def olmo_190m(cls) -> "ModelConfig":
        return ModelConfig(
            compile=True,
            d_model=768,
            n_heads=12,
            n_layers=12,
            rope_theta=500_000,
            flash_attention=True,
            max_sequence_length=4096,
        )

    @classmethod
    def olmo_1b(cls) -> "ModelConfig":
        """
        OLMo-1b (1_336_035_328 parameters)
        """
        return ModelConfig(
            compile=True,
            d_model=2048,
            n_heads=16,
            n_layers=18,
            rope_theta=500_000,
            flash_attention=True,
            max_sequence_length=4096,
        )


class SupportedModels(Enum):
    olmo_190m = ModelConfig.olmo_190m()
    olmo_30m = ModelConfig.olmo_30m()
    olmo_1b = ModelConfig.olmo_1b()


class SupportedTokenizers(Enum):
    dolma2 = TokenizerConfig.dolma2()
    gpt_neox = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
