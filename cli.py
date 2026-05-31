import argparse
from pathlib import Path

from nanolens.training.trainer import trainer, load_model
from nanolens.inference.generate import generate
from nanolens.analysis.inspector import inspect
from nanolens.data.loader import get_device
from nanolens.analysis.visualize import plot_all_heads, plot_hidden_state_norms, plot_cosine_similarity, plot_norm_deltas


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
    
    parser.add_argument("-d", "--dry_run", action="store_true", help =  "Runs a dry run to see everything is working")

    return parser.parse_args()

def dryrun():
    import torch
    from nanolens.model.transformer import TransformerModel
    from nanolens.data.loader import get_batch, get_device
    from nanolens.data.tokenizer import encode, decode, vocab_size
    from nanolens.training.optimizer import build_optim
    from nanolens.analysis.inspector import inspect

    print("Running dryrun checks...\n")
    device = get_device()

    tokens = encode("Raskolnikov")
    assert len(tokens) > 0
    assert decode(tokens) == "Raskolnikov"
    print("✓ Tokenizer — encode/decode roundtrip passed")
    print(f"  vocab_size: {vocab_size}")

    x, y = get_batch('train')
    assert x.shape[1] > 0
    print(f"✓ Loader — batch shape: {x.shape}")

    model = TransformerModel().to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ Model — instantiated, {total:,} parameters")

    with torch.no_grad():
        logits, loss = model(x, y)
    assert logits.shape[-1] == vocab_size
    print(f"✓ Forward pass — logits shape: {logits.shape}, loss: {loss.item():.4f}")

    x_small = torch.zeros((1, 8), dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, weights = model(x_small, return_weights=True)
    assert len(weights) == 8
    print(f"✓ return_weights — {len(weights)} layers captured, shape: {weights[0].shape}")

    optimizer = build_optim(model, learning_rate=3e-4, weight_decay=0.1)
    print(f"✓ Optimizer — {len(optimizer.param_groups)} param groups")

    result = inspect(model, "Raskolnikov", device)
    assert len(result['hidden_states']) == 8
    assert len(result['attention_weights']) == 8
    print(f"✓ Inspector — hidden states: {len(result['hidden_states'])}, attention weights: {len(result['attention_weights'])}")

    print("\nAll checks passed. Everything is wired correctly.")

if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    print(r'''
                 __
                 ||          `7MN.   `7MF'                           `7MMF'
                ====           MMN.    M                               MM 
                |  |__         M YMb   M  ,6"Yb.  `7MMpMMMb.  ,pW"Wq.  MM         .gP"Ya `7MMpMMMb.  ,pP"Ybd
                |  |-.\        M  `MN. M 8)   MM    MM    MM 6W'   `Wb MM        ,M'   Yb  MM    MM  8I   `"
                |__|  \\       M   `MM.M  ,pm9MM    MM    MM 8M     M8 MM      , 8M""""""  MM    MM  `YMMMa.
                 ||   ||       M     YMM 8M   MM    MM    MM YA.   ,A9 MM     ,M YM.    ,  MM    MM  L.   I8
               ======__|     .JML.    YM `Moo9^Yo..JMML  JMML.`Ybmd9'.JMMmmmmMMM  `Mbmmd'.JMML  JMML.M9mmmP'
              ________||__
             /____________\            mechanistic interpretability toolkit · built from scratch
            
          ''')
    
    if args.dry_run:
        dryrun()
    else:
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
                plot_hidden_state_norms(
                    result,
                    positions=[0, 11, 28, 49, 38, 59],
                    labels=['R', 'space1', 'space3', 'space5', ',', '.'],
                    title='Structural Token Norm Trajectories Across Layers',
                    filename='norm_trajectory_structural.png'
                )

                plot_hidden_state_norms(
                    result,
                    positions=[0, 12, 37, 11],
                    labels=['R', 'h', 'd', 'space1'],
                    title='Content Token Norm Trajectories Across Layers',
                    filename='norm_trajectory_content.png'
                )
                plot_norm_deltas(result, positions=[0, 11, 28, 49, 12, 37, 38, 59], labels=['R', 'space1', 'space3', 'space5', 'h', 'd', ',', '.'])
                plot_cosine_similarity(result, positions=[0, 11, 28, 49, 12, 37, 38, 59], labels=['R', 'space1', 'space3', 'space5', 'h', 'd', ',', '.'])

        else:
            generate(model, max_new_tokens=args.tokens)