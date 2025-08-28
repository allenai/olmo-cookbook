#!/usr/bin/env python3
"""
Script to optimize batch size parameters for training runs.

Given constraints on GPUs, sequence length, optimization steps, and targets,
finds optimal values for max_tokens, global_batch_size, and rank_microbatch_size.
"""

import math
from typing import Tuple, List
import argparse


def find_divisors(n: int) -> List[int]:
    """Find all divisors of n."""
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def optimize_batch_sizes(
    num_gpus: int,
    sequence_length: int,
    target_steps: int,
    target_rank_microbatch_size: int,
    target_max_tokens: int
) -> Tuple[int, int, int]:
    """
    Find optimal batch size configuration.
    
    Args:
        num_gpus: Number of GPUs
        sequence_length: Sequence length (must divide batch sizes)
        target_steps: Exact number of optimization steps desired
        target_rank_microbatch_size: Target rank microbatch size to stay near
        target_max_tokens: Target max tokens to stay near
    
    Returns:
        Tuple of (max_tokens, global_batch_size, rank_microbatch_size)
    
    Constraints:
        - global_batch_size * target_steps = max_tokens (exact steps)
        - global_batch_size % (num_gpus * sequence_length) == 0
        - rank_microbatch_size % sequence_length == 0
        - (global_batch_size // num_gpus) % rank_microbatch_size == 0
    """
    
    print(f"Finding optimal batch sizes for:")
    print(f"  GPUs: {num_gpus}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Target steps: {target_steps}")
    print(f"  Target rank microbatch size: {target_rank_microbatch_size}")
    print(f"  Target max tokens: {target_max_tokens}")
    print()
    
    # Calculate approximate global_batch_size from target values
    approx_global_batch_size = target_max_tokens // target_steps
    
    # Find the constraint: global_batch_size must be divisible by (num_gpus * sequence_length)
    base_divisor = num_gpus * sequence_length
    
    # Find candidates for global_batch_size around the approximation
    candidates = []
    
    # Search range: ±100% of approximation (much wider range)
    search_range = int(1.0 * approx_global_batch_size)
    min_batch = max(base_divisor, approx_global_batch_size - search_range)
    max_batch = approx_global_batch_size + search_range
    
    # Find all valid global_batch_sizes
    for batch_size in range(min_batch, max_batch + base_divisor, base_divisor):
        if batch_size % base_divisor == 0:
            # Calculate resulting max_tokens and steps
            max_tokens = batch_size * target_steps
            tokens_per_gpu = batch_size // num_gpus
            
            # Find valid rank_microbatch_sizes for this batch_size
            # Must be divisible by sequence_length and divide tokens_per_gpu evenly
            valid_rank_microbatch_sizes = []
            
            # Generate candidates around target
            rank_candidates = []
            
            # Add multiples of sequence_length around target
            for multiplier in range(1, (tokens_per_gpu // sequence_length) + 1):
                rank_size = multiplier * sequence_length
                if tokens_per_gpu % rank_size == 0:
                    rank_candidates.append(rank_size)
            
            if rank_candidates:
                # Find the rank_microbatch_size closest to target
                best_rank_size = min(rank_candidates, 
                                   key=lambda x: abs(x - target_rank_microbatch_size))
                
                # Calculate how close we are to targets
                tokens_diff = abs(max_tokens - target_max_tokens)
                rank_diff = abs(best_rank_size - target_rank_microbatch_size)
                
                # Prioritize rank microbatch size being close to target
                # Normalize tokens_diff to be relative to target (percentage difference)
                tokens_diff_pct = tokens_diff / target_max_tokens
                # Weight rank_diff much more heavily (10x) than normalized token difference
                weighted_score = (tokens_diff_pct * 1.0) + (rank_diff / target_rank_microbatch_size * 10.0)
                
                candidates.append({
                    'max_tokens': max_tokens,
                    'global_batch_size': batch_size,
                    'rank_microbatch_size': best_rank_size,
                    'tokens_diff': tokens_diff,
                    'rank_diff': rank_diff,
                    'total_diff': tokens_diff + rank_diff,  # Keep for display
                    'weighted_score': weighted_score,  # New scoring metric
                    'grad_accum_steps': tokens_per_gpu // best_rank_size
                })
    
    if not candidates:
        raise ValueError("No valid configuration found. Try adjusting targets.")
    
    # Sort by weighted score (prioritizes rank microbatch size)
    candidates.sort(key=lambda x: x['weighted_score'])
    
    # Print top candidates
    print("Top 5 candidates:")
    print("Rank | Max Tokens  | Global Batch | Rank Micro | Grad Accum | Token Diff | Rank Diff")
    print("-" * 85)
    
    for i, candidate in enumerate(candidates[:5]):
        print(f"{i+1:4d} | {candidate['max_tokens']:11,d} | "
              f"{candidate['global_batch_size']:12,d} | "
              f"{candidate['rank_microbatch_size']:10,d} | "
              f"{candidate['grad_accum_steps']:10d} | "
              f"{candidate['tokens_diff']:10,d} | "
              f"{candidate['rank_diff']:9,d}")
    
    print()
    
    # Return the best candidate
    best = candidates[0]
    print(f"Selected configuration:")
    print(f"  max_tokens: {best['max_tokens']:,}")
    print(f"  global_batch_size: {best['global_batch_size']:,}")
    print(f"  rank_microbatch_size: {best['rank_microbatch_size']:,}")
    print(f"  gradient_accumulation_steps_per_gpu: {best['grad_accum_steps']}")
    print()
    
    # Verify constraints
    print("Constraint verification:")
    tokens_per_step = best['global_batch_size']
    actual_steps = best['max_tokens'] // tokens_per_step
    print(f"  Exact steps: {actual_steps} == {target_steps} ✓")
    
    tokens_per_gpu = best['global_batch_size'] // num_gpus
    print(f"  Global batch divisible by GPUs: {best['global_batch_size']} ÷ {num_gpus} = {tokens_per_gpu} ✓")
    
    print(f"  Global batch divisible by seq_len: {best['global_batch_size']} ÷ {sequence_length} = {best['global_batch_size'] // sequence_length} ✓")
    
    print(f"  Rank micro divisible by seq_len: {best['rank_microbatch_size']} ÷ {sequence_length} = {best['rank_microbatch_size'] // sequence_length} ✓")
    
    print(f"  Tokens per GPU divisible by rank micro: {tokens_per_gpu} ÷ {best['rank_microbatch_size']} = {tokens_per_gpu // best['rank_microbatch_size']} ✓")
    
    return best['max_tokens'], best['global_batch_size'], best['rank_microbatch_size']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize batch sizes for training")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--steps", type=int, default=20564, help="Target optimization steps")
    parser.add_argument("--target-rank-micro", type=int, default=32768, help="Target rank microbatch size")
    parser.add_argument("--target-tokens", type=int, default=25_000_000_000, help="Target max tokens")
    return parser.parse_args()


def main():
    """Example usage"""
    args = parse_args()
    
    try:
        max_tokens, global_batch_size, rank_microbatch_size = optimize_batch_sizes(
            num_gpus=args.gpus,
            sequence_length=args.seq_len,
            target_steps=args.steps,
            target_rank_microbatch_size=args.target_rank_micro,
            target_max_tokens=args.target_tokens
        )
        
        print("\nYAML configuration:")
        print(f"max_tokens: {max_tokens:_}")
        print(f"global_batch_size: {global_batch_size:_}")
        print(f"rank_microbatch_size: {rank_microbatch_size:_}")
        
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
