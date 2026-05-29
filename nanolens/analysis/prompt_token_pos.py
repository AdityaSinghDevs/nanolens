import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from nanolens.analysis.inspector import inspect
from nanolens.training.trainer import load_model
from nanolens.data.loader import get_device
from nanolens.data.tokenizer import decode

device = get_device()
model = load_model("nanolens/checkpoints/nanolens_v1.pt")

prompt = "Raskolnikov hesitated at the threshold, his hands trembling."
result = inspect(model, prompt, device)

print([decode([t]) for t in result['tokens']])