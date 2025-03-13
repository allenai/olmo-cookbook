"""Script to copy a new-style config to run with a larger batch size."""

from argparse import ArgumentParser
import yaml
from math import sqrt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to base config file")
    parser.add_argument("save_path", type=str, help="Path to save new config file")
    parser.add_argument("--name", "-n", type=str, default="code")
    parser.add_argument("--description", type=str, default="Train code model with larger batch size")
    parser.add_argument("--load-path", type=str, default="gs://ai2-llm/checkpoints/ai2-tylerm/olmo-cookbook-1b-5xC-love2code-python-no-prose-hlr-c0c0f2d1/step0/")
    parser.add_argument("--multiplier", "-k", type=float, default=1.)
    parser.add_argument("--max-tokens", "-t", type=int, default=512 * 1024 * 4096)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--budget", type=str, default="ai2/oe-training")
    parser.add_argument("--workspace", type=str, default="ai2/13B")
    parser.add_argument("--nodes", type=int, default=None)
    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    learning_rate = config.get("learning_rate", 1e-4)
    sequence_length = config.get("sequence_length", 2048)
    global_batch_size = config.get("global_batch_size", 1024 * sequence_length)

    config["name"] = args.name
    config["description"] = args.description
    config["max_tokens"] = args.start_step * global_batch_size + args.max_tokens
    config["budget"] = args.budget
    config["workspace"] = args.workspace
    config["load_path"] = args.load_path
    config["global_batch_size"] = int(global_batch_size * args.multiplier)
    config["learning_rate"] = learning_rate * sqrt(args.multiplier)

    if args.nodes is not None:
        config["nodes"] = args.nodes

    with open(args.save_path, "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    main(parse_args())