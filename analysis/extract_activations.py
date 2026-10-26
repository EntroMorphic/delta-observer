#!/usr/bin/env python3
"""
[LEGACY] Extract per-layer activation landscapes from the trained source models.

Runs both trained 4-bit adders over all 512 input cases and saves every
hidden layer's pre/post-activation values. These files are the input to
analysis/geometric_analysis.py.

Requires:
    models/4bit_monolithic_best.pt     (from models/train_4bit_monolithic.py)
    models/4bit_compositional_best.pt  (from models/train_4bit_compositional.py)

Produces:
    data/monolithic_activations.npz
    data/compositional_activations.npz
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.train_4bit_monolithic import Monolithic4BitAdder, generate_dataset
from models.train_4bit_compositional import Compositional4BitAdder


def load_model(model, checkpoint_name):
    """Load a trained checkpoint into model."""
    checkpoint = torch.load(REPO_ROOT / 'models' / checkpoint_name, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def flatten_activations(activations, prefix=''):
    """Flatten nested activation dicts: {'bit0': {'layer1_pre': ...}} -> {'bit0_layer1_pre': ...}."""
    flat = {}
    for key, value in activations.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_activations(value, prefix=f"{name}_"))
        else:
            flat[name] = value
    return flat


def main():
    print("=" * 80)
    print("ACTIVATION EXTRACTION: 4-BIT ADDERS")
    print("=" * 80)

    inputs, outputs = generate_dataset()
    inputs = inputs.float()
    outputs = outputs.float()

    models = [
        ('monolithic', load_model(Monolithic4BitAdder(), '4bit_monolithic_best.pt')),
        ('compositional', load_model(Compositional4BitAdder(), '4bit_compositional_best.pt')),
    ]

    data_dir = REPO_ROOT / 'data'
    data_dir.mkdir(exist_ok=True)

    for name, model in models:
        with torch.no_grad():
            _, activations = model(inputs, return_activations=True)
        layers = flatten_activations(activations)

        save_path = data_dir / f'{name}_activations.npz'
        np.savez(save_path, inputs=inputs.numpy(), outputs=outputs.numpy(), **layers)
        print(f"💾 Saved {save_path.relative_to(REPO_ROOT)} ({len(layers)} layers)")

    print("\n✅ Ready for analysis/geometric_analysis.py")


if __name__ == "__main__":
    main()
