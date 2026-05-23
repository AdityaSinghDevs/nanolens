import argparse
from pathlib import Path
from nanolens.training.trainer import trainer, load_model
from nanolens.inference.generate import generate

CHECKPOINT = Path("checkpoints/model.pt")


def parse_args():
    parser = argparse.ArgumentParser(description = "Nanolens")

    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to checkpoint file e.g. checkpoints/gpt_L4_H4_E32_5000.pt")

    parser.add_argument("-g", "--generate", action= "store_true", help= "Generate (requires --checkpoint)")

    parser.add_argument("-t", "--tokens", type= int, default = 500, help = "Number of tokens to generate (default: 500)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.checkpoint:
        model = load_model(args.checkpoint)
    else:
        model = trainer()
    
    generate(model, max_new_tokens = args.tokens)