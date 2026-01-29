"""
Sample training data from a shuffled FSL dataset.

This script is designed to run remotely on Beaker (without GPU) to sample
training instances from a shuffled FSL dataset, decode them using the
HuggingFace tokenizer, and output both the decoded text and document metadata.

This version instruments the actual data pipeline rather than mimicking it,
ensuring the document metadata exactly matches what the training sees.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import numpy as np
import torch
from olmo_core.data.numpy_dataset import NumpyShuffledFSLDataset
from olmo_core.data.utils import load_array_slice_into_tensor
from transformers import AutoTokenizer

from cookbook.utils.config import build_dataset_only

logger = logging.getLogger(__name__)


class InstrumentedShuffledFSLDataset:
    """
    A wrapper around NumpyShuffledFSLDataset that captures document metadata
    during sampling by instrumenting the actual __getitem__ logic.
    
    This ensures we get exactly the same document information that the
    training pipeline sees, avoiding any discrepancies from reimplementing
    the sampling logic.
    """
    
    def __init__(self, dataset: NumpyShuffledFSLDataset):
        self.dataset = dataset
        self._last_documents: List[Dict[str, Any]] = []
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    @property
    def sequence_length(self) -> int:
        return self.dataset.sequence_length
    
    def get_instance_with_metadata(self, index: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get an instance along with metadata about the documents that compose it.
        
        This reimplements __getitem__ but captures document info as it goes,
        using the exact same logic as the original to ensure consistency.
        
        Returns:
            Tuple of (instance_dict, documents_list) where:
            - instance_dict: {"input_ids": tensor} same as original __getitem__
            - documents_list: list of document metadata dicts
        """
        dataset = self.dataset
        
        index = int(index)
        pos_index = index if index >= 0 else len(dataset) + index
        
        if pos_index < 0 or pos_index >= len(dataset):
            raise IndexError(f"{index} is out of bounds for dataset of size {len(dataset)}")
        
        # Ensure index is loaded
        dataset._load_or_build_index()
        assert dataset._cumsum is not None
        assert dataset._permutation is not None
        assert dataset._doc_metadata is not None
        
        start_token = pos_index * dataset.sequence_length
        end_token = start_token + dataset.sequence_length
        
        # Binary search: find the documents that span this token range
        first_doc = int(np.searchsorted(dataset._cumsum, start_token, side="right")) - 1
        last_doc = int(np.searchsorted(dataset._cumsum, end_token, side="left"))
        
        # Clamp to valid range
        first_doc = max(0, first_doc)
        last_doc = min(last_doc, len(dataset._permutation) - 1)
        
        tokens_list: List[torch.Tensor] = []
        documents: List[Dict[str, Any]] = []
        
        for shuffled_idx in range(first_doc, last_doc + 1):
            orig_idx = int(dataset._permutation[shuffled_idx])
            file_idx, doc_start, orig_len, proc_len = dataset._doc_metadata[orig_idx]
            
            # Read and process this document (same as original)
            doc_tokens = dataset._read_document(int(file_idx), int(doc_start), int(orig_len), int(proc_len))
            
            # Calculate which portion of this document we need
            doc_start_in_stream = int(dataset._cumsum[shuffled_idx])
            local_start = max(0, start_token - doc_start_in_stream)
            local_end = min(int(proc_len), end_token - doc_start_in_stream)
            
            # This is the key check from the original code - only include docs
            # that actually contribute tokens to this instance
            if local_start < local_end:
                tokens_list.append(doc_tokens[local_start:local_end])
                
                # Capture document metadata for this contributing document
                documents.append({
                    "shuffled_idx": shuffled_idx,
                    "orig_idx": orig_idx,
                    "file_idx": int(file_idx),
                    "file_path": str(dataset.paths[int(file_idx)]),
                    "doc_start": int(doc_start),
                    "orig_len": int(orig_len),
                    "proc_len": int(proc_len),
                    "local_start": local_start,
                    "local_end": local_end,
                    "tokens_contributed": local_end - local_start,
                })
        
        input_ids = torch.cat(tokens_list) if tokens_list else torch.tensor([], dtype=torch.long)
        
        return {"input_ids": input_ids}, documents


def read_full_document(
    dataset,
    file_idx: int,
    doc_start: int,
    orig_len: int,
) -> torch.Tensor:
    """Read the full original document tokens from the source file."""
    path = dataset.paths[file_idx]
    return load_array_slice_into_tensor(path, doc_start, doc_start + orig_len, dataset.dtype)


def sample_and_decode_instance(
    instrumented_dataset: InstrumentedShuffledFSLDataset,
    tokenizer,
    instance_idx: int,
) -> Dict[str, Any]:
    """
    Sample a single instance and decode it with full document information.
    
    Returns a dict containing:
    - instance_idx: the instance index
    - detokenized_instance: the full decoded training instance
    - token_ids: list of token IDs
    - documents: list of document info with full text
    """
    # Get the instance tokens AND document metadata in one call
    instance, doc_infos = instrumented_dataset.get_instance_with_metadata(instance_idx)
    input_ids = instance["input_ids"]
    
    # Detokenize the whole instance
    detokenized_instance = tokenizer.decode(input_ids.tolist())
    
    # Read and decode full documents
    documents = []
    for doc_info in doc_infos:
        full_doc_tokens = read_full_document(
            instrumented_dataset.dataset,
            doc_info["file_idx"],
            doc_info["doc_start"],
            doc_info["orig_len"],
        )
        full_text = tokenizer.decode(full_doc_tokens.tolist())
        
        documents.append({
            **doc_info,
            "full_text": full_text,
        })
    
    return {
        "instance_idx": instance_idx,
        "detokenized_instance": detokenized_instance,
        "token_ids": input_ids.tolist(),
        "documents": documents,
    }


def log_instance(sample: Dict[str, Any]) -> None:
    """Log a sampled instance to the console with clear formatting."""
    logger.info(f"\n{'='*60}")
    logger.info(f"INSTANCE {sample['instance_idx']}")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nDetokenized instance ({len(sample['token_ids'])} tokens):")
    logger.info("-" * 40)
    logger.info(sample["detokenized_instance"])
    logger.info("-" * 40)
    
    logger.info(f"\n--- Documents in this instance ({len(sample['documents'])} documents) ---")
    
    for i, doc in enumerate(sample["documents"]):
        logger.info(f"\n[DOC {i}] shuffled_idx={doc['shuffled_idx']}, orig_idx={doc['orig_idx']}, "
                   f"file_idx={doc['file_idx']}, orig_len={doc['orig_len']}, proc_len={doc['proc_len']}")
        logger.info(f"  File: {doc['file_path']}")
        logger.info(f"  Portion in instance: tokens [{doc['local_start']}:{doc['local_end']}] "
                   f"({doc['tokens_contributed']} tokens)")
        logger.info(f"\nFull document text ({doc['orig_len']} tokens):")
        logger.info("-" * 40)
        logger.info(doc["full_text"])
        logger.info("-" * 40)
    
    logger.info(f"\n{'='*60}\n")


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config-path",
    "-C",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment configuration file.",
)
@click.option(
    "--num-samples",
    "-n",
    type=int,
    default=10,
    help="Number of instances to sample.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(),
    default="/results/samples.jsonl",
    help="Path to write the output JSONL file.",
)
@click.option(
    "--start-idx",
    "-s",
    type=int,
    default=0,
    help="Starting instance index to sample from.",
)
def sample(
    config_path: str,
    num_samples: int,
    output_path: str,
    start_idx: int,
):
    """Sample training data from a shuffled FSL dataset."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info(f"Loading config from {config_path}")
    logger.info(f"Will sample {num_samples} instances starting from index {start_idx}")
    
    # Build dataset and get tokenizer info
    dataset, tokenizer_identifier = build_dataset_only(Path(config_path))
    
    # Wrap with instrumentation
    instrumented_dataset = InstrumentedShuffledFSLDataset(dataset)
    
    logger.info(f"Dataset built with {len(instrumented_dataset)} instances")
    logger.info(f"Sequence length: {instrumented_dataset.sequence_length}")
    logger.info(f"Loading tokenizer: {tokenizer_identifier}")
    
    # Load the HuggingFace tokenizer for decoding
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier)
    
    # Sample instances
    samples = []
    for i in range(num_samples):
        instance_idx = start_idx + i
        
        if instance_idx >= len(instrumented_dataset):
            logger.warning(f"Instance index {instance_idx} exceeds dataset size {len(instrumented_dataset)}, stopping.")
            break
        
        logger.info(f"Sampling instance {instance_idx} ({i + 1}/{num_samples})...")
        
        sample_data = sample_and_decode_instance(instrumented_dataset, tokenizer, instance_idx)
        samples.append(sample_data)
        
        # Log to console
        log_instance(sample_data)
    
    # Write to output file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing {len(samples)} samples to {output_path}")
    
    with open(output_file, "w") as f:
        for sample_data in samples:
            f.write(json.dumps(sample_data) + "\n")
    
    logger.info("Done!")


if __name__ == "__main__":
    cli()
