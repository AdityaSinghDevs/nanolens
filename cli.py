import argparse
from pathlib import Path

from nanolens.training.trainer import trainer, load_model
from nanolens.inference.generate import generate
from nanolens.analysis.inspector import inspect
from nanolens.data.loader import get_device
from nanolens.analysis.visualize import plot_all_heads, plot_hidden_state_norms


CHECKPOINT = Path("checkpoints/model.pt")


def parse_args():
    parser = argparse.ArgumentParser(description = "Nanolens")

    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to checkpoint file e.g. checkpoints/gpt_L4_H4_E32_5000.pt")

    parser.add_argument("-g", "--generate", action= "store_true", help= "Generate (requires --checkpoint)")

    parser.add_argument("-t", "--tokens", type= int, default = 500, help = "Number of tokens to generate (default: 500)")

    parser.add_argument("-i", "--inspect", type=str, default = None, help = 'Prompt to inspect e.g. --inspect "To be or not"')

    parser.add_argument("-aw","--attention_weights", action = "store_true", help=" If you need attention score plots (use along with -i)" )

    parser.add_argument("-hs", "--hidden_states", action = "store_true", help = " If you want to see hidden state norms plotted (use along with -i)")

    parser.add_argument("--resume", type=int, default=0,
                    help="Iter to resume from e.g. --resume 3000")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    if args.resume:
        model = trainer(start_iter=args.resume, checkpoint_path=args.checkpoint)

    elif args.checkpoint:
        model = load_model(args.checkpoint)

    else:
        model = trainer()

    if args.inspect:
        result = inspect(model, args.inspect, device)

        print(f"Tokens: {len(result['tokens'])}")
        print(f"Layers captured : {len(result['attention_weights'])}")
        print(f"Layer 0 shape: {result['attention_weights'][0].shape}")

        if args.attention_weights:
            plot_all_heads(result)
        if args.hidden_states:
            plot_hidden_state_norms(result, positions=[0,11,12,38,59], labels=['R','space','h',',','.'])

    else:
        generate(model, max_new_tokens=args.tokens)