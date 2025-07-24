import os
import torch
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoConfig
import logging

logger = logging.getLogger(__name__)

class ModelMerger:
    
    def __init__(self, checkpoint_paths: List[Union[str, Path]], output_dir: str, 
                 alpha: float = 0.2, force_safetensors: bool = None):
        """
        Initialize the ModelMerger.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints (can be GCP/Weka/local paths)
            output_dir: Directory to save merged models
            alpha: Smoothing factor for EMA (default: 0.2)
            force_safetensors: Force output format (True=safetensors, False=pytorch, None=auto-detect)
        """
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.output_dir = Path(output_dir)
        self.alpha = alpha 
        self.force_safetensors = force_safetensors
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if len(self.checkpoint_paths) < 2:
            raise ValueError(f"Need at least 2 checkpoints for merging, got {len(self.checkpoint_paths)}")
        
    def validate_checkpoints(self) -> List[Path]:
        """Validate that all checkpoint paths exist and contain model files."""
        import glob
        valid_checkpoints = []
        
        for checkpoint_path in self.checkpoint_paths:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
            
            if not checkpoint_path.is_dir():
                raise ValueError(f"Checkpoint path must be a directory: {checkpoint_path}")
                
            has_model = (
                (checkpoint_path / "pytorch_model.bin").exists() or
                (checkpoint_path / "model.safetensors").exists() or
                len(glob.glob(str(checkpoint_path / "pytorch_model-*.bin"))) > 0 or
                len(glob.glob(str(checkpoint_path / "model-*-of-*.safetensors"))) > 0
            )
            
            if not has_model:
                raise FileNotFoundError(f"No model files found in {checkpoint_path}")
                
            valid_checkpoints.append(checkpoint_path)
            
        logger.info(f"Validated {len(valid_checkpoints)} checkpoints: {[cp.name for cp in valid_checkpoints]}")
        return valid_checkpoints
    
    def load_state_dict(self, checkpoint_path: Path) -> Dict[str, torch.Tensor]:
        from safetensors.torch import load_file
        import glob
        safetensors_shards = glob.glob(str(checkpoint_path / "model-*-of-*.safetensors"))
        if safetensors_shards:
            logger.info(f"Loading sharded safetensors from {len(safetensors_shards)} files")
            state_dict = {}
            for shard_file in sorted(safetensors_shards):
                shard_dict = load_file(shard_file)
                state_dict.update(shard_dict)
                logger.debug(f"Loaded shard: {Path(shard_file).name}")
            return state_dict
        safetensors_path = checkpoint_path / "model.safetensors"
        if safetensors_path.exists():
            logger.info(f"Loading single safetensors from {safetensors_path}")
            return load_file(safetensors_path)
        pytorch_path = checkpoint_path / "pytorch_model.bin"
        if pytorch_path.exists():
            logger.info(f"Loading pytorch model from {pytorch_path}")
            return torch.load(pytorch_path, map_location='cpu')
        pytorch_shards = glob.glob(str(checkpoint_path / "pytorch_model-*.bin"))
        if pytorch_shards:
            logger.info(f"Loading sharded pytorch model from {len(pytorch_shards)} files")
            state_dict = {}
            for shard_file in sorted(pytorch_shards):
                shard_dict = torch.load(shard_file, map_location='cpu')
                state_dict.update(shard_dict)
                logger.debug(f"Loaded shard: {Path(shard_file).name}")
            return state_dict
        
        raise FileNotFoundError(f"No model files found in {checkpoint_path}. Expected formats: "
                              f"model-*-of-*.safetensors, model.safetensors, pytorch_model.bin, or pytorch_model-*.bin")
    
    def save_state_dict(self, state_dict: Dict[str, torch.Tensor], output_path: Path, 
                       reference_checkpoint: Path, suffix: str, use_safetensors: bool = True):
        output_path.mkdir(parents=True, exist_ok=True)
        if use_safetensors:
            try:
                from safetensors.torch import save_file
                save_file(state_dict, output_path / "model.safetensors")
                logger.info(f"Saved model.safetensors to {output_path}")
            except ImportError:
                logger.warning("safetensors not available, falling back to pytorch format")
                torch.save(state_dict, output_path / "pytorch_model.bin")
                logger.info(f"Saved pytorch_model.bin to {output_path}")
        else:
            torch.save(state_dict, output_path / "pytorch_model.bin")
            logger.info(f"Saved pytorch_model.bin to {output_path}")
        files_to_copy = [
            "config.json", 
            "tokenizer_config.json", 
            "tokenizer.json",
            "special_tokens_map.json",
            "vocab.txt",
            "merges.txt",
            "added_tokens.json",
            "generation_config.json"
        ]
        
        for filename in files_to_copy:
            src = reference_checkpoint / filename
            dst = output_path / filename
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                logger.debug(f"Copied {filename}")
        config_path = output_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            config['_merge_info'] = {
                'method': suffix,
                'num_checkpoints': len(self.checkpoint_paths),
                'checkpoint_paths': [str(p) for p in self.checkpoint_paths],
                'alpha': self.alpha if suffix == 'ema' else None
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        logger.info(f"Saved merged model to {output_path}")
    
    def simple_moving_average(self, state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Compute Simple Moving Average of model weights."""
        logger.info("Computing Simple Moving Average (SMA)")
        N = len(state_dicts)
        merged_dict = {}
        for key in state_dicts[0].keys():
            merged_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict in state_dicts:
            for key in merged_dict.keys():
                merged_dict[key] += state_dict[key]
        for key in merged_dict.keys():
            merged_dict[key] /= N
            
        return merged_dict
    
    def weighted_moving_average(self, state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        logger.info("Computing Weighted Moving Average (WMA)")
        N = len(state_dicts)
        weights = [i + 1 for i in range(N)]  # [1, 2, 3, 4, 5, ...]
        weight_sum = sum(weights)
        merged_dict = {}
        for key in state_dicts[0].keys():
            merged_dict[key] = torch.zeros_like(state_dicts[0][key])
        for i, state_dict in enumerate(state_dicts):
            weight = weights[i] / weight_sum
            logger.debug(f"Checkpoint {i+1} weight: {weight:.4f}")
            for key in merged_dict.keys():
                merged_dict[key] += weight * state_dict[key]
                
        return merged_dict
    
    def exponential_moving_average(self, state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Compute Exponential Moving Average with specified alpha."""
        logger.info(f"Computing Exponential Moving Average (EMA) with α={self.alpha}")
        N = len(state_dicts)
        
        weights = []
        for i in range(N):
            weight = self.alpha * ((1 - self.alpha) ** (N - 1 - i))
            weights.append(weight)
        
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        merged_dict = {}

        for key in state_dicts[0].keys():
            merged_dict[key] = torch.zeros_like(state_dicts[0][key])
    
        for i, state_dict in enumerate(state_dicts):
            weight = weights[i]
            logger.debug(f"Checkpoint {i+1} weight: {weight:.4f}")
            for key in merged_dict.keys():
                merged_dict[key] += weight * state_dict[key]
                
        return merged_dict
    
    def merge_models(self, methods: List[str] = None):
        """
        Merge checkpoints using specified methods.
        
        Args:
            methods: List of methods to use. Options: ['sma', 'wma', 'ema']
                    If None, uses all methods.
        """
        if methods is None:
            methods = ['sma', 'wma', 'ema']
            
        valid_checkpoints = self.validate_checkpoints()
        import glob
        first_checkpoint = valid_checkpoints[0]
        input_uses_safetensors = (
            len(glob.glob(str(first_checkpoint / "model-*-of-*.safetensors"))) > 0 or
            (first_checkpoint / "model.safetensors").exists()
        )
        
        if self.force_safetensors is not None:
            use_safetensors = self.force_safetensors
            logger.info(f"Output format forced to: {'safetensors' if use_safetensors else 'pytorch'}")
        else:
            use_safetensors = input_uses_safetensors
            logger.info(f"Using {'safetensors' if use_safetensors else 'pytorch'} format (auto-detected from input)")
        
        logger.info("Loading all state dictionaries...")
        state_dicts = []
        for checkpoint in valid_checkpoints:
            state_dict = self.load_state_dict(checkpoint)
            state_dicts.append(state_dict)
            logger.info(f"Loaded {checkpoint.name}: {len(state_dict)} parameters")
        
        reference_checkpoint = valid_checkpoints[0]  
    
        method_funcs = {
            'sma': self.simple_moving_average,
            'wma': self.weighted_moving_average, 
            'ema': self.exponential_moving_average
        }
        
        for method_name in methods:
            if method_name not in method_funcs:
                logger.warning(f"Unknown method '{method_name}', skipping. Available: {list(method_funcs.keys())}")
                continue
                
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {method_name.upper()} merge")
            logger.info(f"{'='*50}")
            
            method_func = method_funcs[method_name]
            merged_dict = method_func(state_dicts)
        
            output_path = self.output_dir / f"merged_{method_name}"
            self.save_state_dict(merged_dict, output_path, reference_checkpoint, method_name, use_safetensors)
            
            total_params = sum(p.numel() for p in merged_dict.values())
            logger.info(f"Merged model has {total_params:,} parameters")
        
        logger.info(f"\n{'='*50}")
        logger.info("All merging methods completed successfully!")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"{'='*50}")