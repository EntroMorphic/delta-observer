#!/usr/bin/env python3
"""
Prepare dataset for Delta Observer training.

Extract activations from both trained models and compute semantic labels (carry_count).
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('/home/ubuntu/geometric-microscope')

from models.train_4bit_monolithic import Monolithic4BitAdder
from models.train_4bit_compositional import Compositional4BitAdder


def load_models():
    """Load trained models."""
    print("Loading trained models...")
    
    # Monolithic
    mono_model = Monolithic4BitAdder()
    mono_checkpoint = torch.load('/home/ubuntu/geometric-microscope/models/4bit_monolithic_best.pt')
    mono_model.load_state_dict(mono_checkpoint['model_state_dict'])
    mono_model.eval()
    
    # Compositional
    comp_model = Compositional4BitAdder()
    comp_checkpoint = torch.load('/home/ubuntu/geometric-microscope/models/4bit_compositional_best.pt')
    comp_model.load_state_dict(comp_checkpoint['model_state_dict'])
    comp_model.eval()
    
    print("✅ Models loaded")
    return mono_model, comp_model


def generate_all_inputs():
    """Generate all 512 possible 4-bit additions."""
    inputs = []
    for a in range(16):
        for b in range(16):
            for carry_in in [0, 1]:
                # Convert to binary
                a_bits = [(a >> i) & 1 for i in range(4)]
                b_bits = [(b >> i) & 1 for i in range(4)]
                
                input_vec = a_bits + b_bits + [carry_in]
                inputs.append(input_vec)
    
    return torch.tensor(inputs, dtype=torch.float32)


def compute_carry_count(a_bits, b_bits, carry_in):
    """
    Count how many carry operations occur during addition.
    
    This is the semantic variable we'll use to test if the Delta Observer
    learned meaningful structure.
    """
    carries = 0
    carry = carry_in
    
    for i in range(4):
        bit_sum = a_bits[i] + b_bits[i] + carry
        if bit_sum >= 2:
            carries += 1
            carry = 1
        else:
            carry = 0
    
    return carries


def extract_activations(model, inputs, model_type='monolithic'):
    """
    Extract hidden layer activations from model.
    
    For monolithic: single hidden layer (64 neurons)
    For compositional: all 4 bit-adder hiddens concatenated (4 × 8 = 32 neurons)
    """
    activations = []
    
    with torch.no_grad():
        for input_vec in inputs:
            input_vec = input_vec.unsqueeze(0)  # Add batch dimension
            
            if model_type == 'monolithic':
                # Extract from hidden layer 1 (after first linear + relu)
                x = model.net[0](input_vec)  # Linear
                x = model.net[1](x)  # ReLU
                activations.append(x.squeeze(0).numpy())
            
            elif model_type == 'compositional':
                # Extract from all 4 bit-adders' hidden layers
                # Need to properly construct inputs for each bit adder
                
                # Extract bits from input
                a = input_vec[:, 0:4]  # [a0, a1, a2, a3]
                b = input_vec[:, 4:8]  # [b0, b1, b2, b3]
                carry_in = input_vec[:, 8:9]  # [carry_in]
                
                bit_acts = []
                
                # Bit 0: a0, b0, carry_in
                bit0_input = torch.cat([a[:, 0:1], b[:, 0:1], carry_in], dim=1)
                x = model.bit0_adder.net[0](bit0_input)
                x = model.bit0_adder.net[1](x)
                bit_acts.append(x.squeeze(0).numpy())
                
                # For bits 1-3, we need the carry from previous bit
                # But we're just extracting activations, so use the actual model forward
                # to get proper carries
                with torch.no_grad():
                    # Get carries by running full forward pass
                    bit0_output = model.bit0_adder(bit0_input)
                    c0 = bit0_output[:, 1:2]
                    
                    bit1_input = torch.cat([a[:, 1:2], b[:, 1:2], c0], dim=1)
                    x = model.bit1_adder.net[0](bit1_input)
                    x = model.bit1_adder.net[1](x)
                    bit_acts.append(x.squeeze(0).numpy())
                    
                    bit1_output = model.bit1_adder(bit1_input)
                    c1 = bit1_output[:, 1:2]
                    
                    bit2_input = torch.cat([a[:, 2:3], b[:, 2:3], c1], dim=1)
                    x = model.bit2_adder.net[0](bit2_input)
                    x = model.bit2_adder.net[1](x)
                    bit_acts.append(x.squeeze(0).numpy())
                    
                    bit2_output = model.bit2_adder(bit2_input)
                    c2 = bit2_output[:, 1:2]
                    
                    bit3_input = torch.cat([a[:, 3:4], b[:, 3:4], c2], dim=1)
                    x = model.bit3_adder.net[0](bit3_input)
                    x = model.bit3_adder.net[1](x)
                    bit_acts.append(x.squeeze(0).numpy())
                
                # Concatenate all bit-adder activations
                concat_act = np.concatenate(bit_acts)
                activations.append(concat_act)
    
    return np.array(activations)


def prepare_dataset():
    """
    Prepare complete dataset for Delta Observer training.
    """
    print("="*80)
    print("DELTA OBSERVER DATASET PREPARATION")
    print("="*80)
    
    # Load models
    mono_model, comp_model = load_models()
    
    # Generate all inputs
    print("\n📊 Generating all 512 inputs...")
    inputs = generate_all_inputs()
    print(f"   Shape: {inputs.shape}")
    
    # Extract activations
    print("\n🔬 Extracting monolithic activations...")
    mono_activations = extract_activations(mono_model, inputs, 'monolithic')
    print(f"   Shape: {mono_activations.shape}")
    
    print("\n🔬 Extracting compositional activations...")
    comp_activations = extract_activations(comp_model, inputs, 'compositional')
    print(f"   Shape: {comp_activations.shape}")
    
    # Compute semantic labels
    print("\n🏷️  Computing semantic labels...")
    carry_counts = []
    bit_positions = []  # Which bit position has highest activation
    
    for i, input_vec in enumerate(inputs):
        # Extract a, b, carry_in
        a_bits = input_vec[:4].numpy()
        b_bits = input_vec[4:8].numpy()
        carry_in = input_vec[8].item()
        
        # Compute carry count
        carry_count = compute_carry_count(a_bits, b_bits, carry_in)
        carry_counts.append(carry_count)
        
        # Determine which bit position has highest activation in compositional
        # (This is the "label" for classification)
        bit_acts = [comp_activations[i, j*8:(j+1)*8].sum() for j in range(4)]
        bit_pos = np.argmax(bit_acts)
        bit_positions.append(bit_pos)
    
    carry_counts = np.array(carry_counts)
    bit_positions = np.array(bit_positions)
    
    print(f"   Carry count range: {carry_counts.min()}-{carry_counts.max()}")
    print(f"   Carry count distribution: {np.bincount(carry_counts)}")
    print(f"   Bit position distribution: {np.bincount(bit_positions)}")
    
    # Compute outputs for analysis
    print("\n🧮 Computing outputs...")
    outputs = []
    with torch.no_grad():
        for input_vec in inputs:
            output = mono_model(input_vec.unsqueeze(0))
            outputs.append(output.squeeze(0).numpy())
    outputs = np.array(outputs)
    
    # Save dataset
    print("\n💾 Saving dataset...")
    np.savez(
        '/home/ubuntu/geometric-microscope/analysis/delta_observer_dataset.npz',
        inputs=inputs.numpy(),
        mono_activations=mono_activations,
        comp_activations=comp_activations,
        carry_counts=carry_counts,
        bit_positions=bit_positions,
        outputs=outputs,
    )
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE")
    print("="*80)
    print("\n📊 Dataset summary:")
    print(f"   Samples: {len(inputs)}")
    print(f"   Monolithic activations: {mono_activations.shape}")
    print(f"   Compositional activations: {comp_activations.shape}")
    print(f"   Carry counts: {carry_counts.shape}")
    print(f"   Bit positions: {bit_positions.shape}")
    print(f"\n✅ Ready for Delta Observer training!")


if __name__ == "__main__":
    prepare_dataset()
