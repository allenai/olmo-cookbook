"""
Sample training data from a shuffled FSL dataset.

This script is designed to run remotely on Beaker (without GPU) to sample
training instances from a shuffled FSL dataset, decode them using the
HuggingFace tokenizer, and output both the decoded text and document metadata.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import click
import numpy as np
import torch
from olmo_core.data.utils import load_array_slice_into_tensor
from transformers import AutoTokenizer

from cookbook.utils.config import build_dataset_only

logger = logging.getLogger(__name__)


def get_documents_for_instance(
    dataset,
    instance_idx: int,
) -> List[Dict[str, Any]]:
    """
    Get document metadata for a given instance index.
    
    Returns a list of document info dicts containing:
    - shuffled_idx: position in shuffled order
    - orig_idx: original document index
    - file_idx: source file index
    - file_path: path to source file
    - doc_start: start position in file
    - orig_len: original document length
    - proc_len: processed length (after truncation)
    - local_start: start offset within document for this instance
    - local_end: end offset within document for this instance
    """
    # Ensure index is loaded
    dataset._load_or_build_index()
    
    assert dataset._cumsum is not None
    assert dataset._permutation is not None
    assert dataset._doc_metadata is not None
    
    sequence_length = dataset.sequence_length
    start_token = instance_idx * sequence_length
    end_token = start_token + sequence_length
    
    # Binary search to find documents spanning this token range
    first_doc = int(np.searchsorted(dataset._cumsum, start_token, side="right")) - 1
    last_doc = int(np.searchsorted(dataset._cumsum, end_token, side="left"))
    
    # Clamp to valid range
    first_doc = max(0, first_doc)
    last_doc = min(last_doc, len(dataset._permutation) - 1)
    
    documents = []
    for shuffled_idx in range(first_doc, last_doc + 1):
        orig_idx = int(dataset._permutation[shuffled_idx])
        file_idx, doc_start, orig_len, proc_len = dataset._doc_metadata[orig_idx]
        
        # Calculate which portion of this document is in this instance
        doc_start_in_stream = int(dataset._cumsum[shuffled_idx])
        local_start = max(0, start_token - doc_start_in_stream)
        local_end = min(int(proc_len), end_token - doc_start_in_stream)
        
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
        })
    
    return documents


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
    dataset,
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
    # Get the instance tokens
    instance = dataset[instance_idx]
    input_ids = instance["input_ids"]
    
    # Detokenize the whole instance
    detokenized_instance = tokenizer.decode(input_ids.tolist())
    
    # Get document metadata
    doc_infos = get_documents_for_instance(dataset, instance_idx)
    
    # Read and decode full documents
    documents = []
    for doc_info in doc_infos:
        full_doc_tokens = read_full_document(
            dataset,
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
    
    logger.info(f"\n--- Document IDs in this instance ({len(sample['documents'])} documents) ---")
    
    for i, doc in enumerate(sample["documents"]):
        logger.info(f"\n[DOC {i}] shuffled_idx={doc['shuffled_idx']}, orig_idx={doc['orig_idx']}, "
                   f"file_idx={doc['file_idx']}, orig_len={doc['orig_len']}, proc_len={doc['proc_len']}")
        logger.info(f"  File: {doc['file_path']}")
        logger.info(f"  Portion used in instance: tokens [{doc['local_start']}:{doc['local_end']}]")
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
    
    logger.info(f"Dataset built with {len(dataset)} instances")
    logger.info(f"Sequence length: {dataset.sequence_length}")
    logger.info(f"Loading tokenizer: {tokenizer_identifier}")
    
    # Load the HuggingFace tokenizer for decoding
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier)
    
    # Sample instances
    samples = []
    for i in range(num_samples):
        instance_idx = start_idx + i
        
        if instance_idx >= len(dataset):
            logger.warning(f"Instance index {instance_idx} exceeds dataset size {len(dataset)}, stopping.")
            break
        
        logger.info(f"Sampling instance {instance_idx} ({i + 1}/{num_samples})...")
        
        sample_data = sample_and_decode_instance(dataset, tokenizer, instance_idx)
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
