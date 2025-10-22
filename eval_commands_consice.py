#!/usr/bin/env python3
"""
Script to print out evaluation commands for a specific gs:// checkpoint path.
Usage: python eval_commands.py <gs://path/to/checkpoint>
"""

import sys
import os

def print_eval_commands(gs_path):
    if not gs_path.startswith("gs://"):
        print("Error: Path must start with gs://", file=sys.stderr)
        sys.exit(1)
    
    # Extract the checkpoint name from the gs path
    # e.g., gs://ai2-llm/checkpoints/ianm/suffix-train-olmo2-5xC-30m-dense-falcon-1fd53d15/step20564
    path_parts = gs_path.split("/")
    checkpoint_dir = "/".join(path_parts[3:])  # Remove gs://ai2-llm
    weka_path = f"/oe-training-default/ai2-llm/{checkpoint_dir}"
    hf_path = f"{weka_path}-hf"
    
    print("### Evaluation Commands ###")
    print()
    print("# Step 1: Copy checkpoint from GCS to Weka")
    print(f'python -m cookbook.remote {gs_path} weka://oe-training-default/ai2-llm/{checkpoint_dir}')
    print()
    
    print("# Step 2: Evaluate on paloma:all (all Paloma corpora)")
    print(f'''olmo-cookbook-eval evaluate \\
  "{weka_path}" \\
  --tasks paloma_falcon-refinedweb::paloma \\
  --priority normal \\
  --cluster aus80g \\
  --partition-size 1 \\
  --num-gpus 1 \\
  --model-backend olmo_core \\
  --gantry-args hf_token=true \\
  --dashboard ianm-suffix-train \\
  --workspace ai2/olmo-3-microanneals''')

def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_commands.py <gs://path/to/checkpoint>")
        print("Example: python eval_commands.py gs://ai2-llm/checkpoints/ianm/suffix-train-olmo2-5xC-30m-dense-falcon-1fd53d15/step20564")
        sys.exit(1)
    
    gs_path = sys.argv[1]
    print_eval_commands(gs_path)

if __name__ == "__main__":
    main()
