import argparse
from pathlib import Path

from nanolens.training.trainer import trainer, load_model
from nanolens.inference.generate import generate
from nanolens.analysis.inspector import inspect
from nanolens.data.loader import get_device
from nanolens.analysis.visualize import plot_attention_head


CHECKPOINT = Path("checkpoints/model.pt")


def parse_args():
    parser = argparse.ArgumentParser(description = "Nanolens")

    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to checkpoint file e.g. checkpoints/gpt_L4_H4_E32_5000.pt")

    parser.add_argument("-g", "--generate", action= "store_true", help= "Generate (requires --checkpoint)")

    parser.add_argument("-t", "--tokens", type= int, default = 500, help = "Number of tokens to generate (default: 500)")

    parser.add_argument("-i", "--inspect", type=str, default = None, help = 'Prompt to inspect e.g. --inspect "To be or not"')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device  = get_device()

    if args.checkpoint:
        model = load_model(args.checkpoint)
    else:
        model = trainer()
    
    if args.inspect:
        result = inspect(model, args.inspect, device)
        print("Hidden states:", list(result['hidden_states'].keys()))

        print("block_0 shape:", result['hidden_states']['block_0'].shape)
        print("Attention layers:", len(result['attention_weights']))
        print("Layer 0 weights shape:", result['attention_weights'][0].shape)
        plot_attention_head(result, layer=7, head=0)
    else:
        generate(model, max_new_tokens = args.tokens)