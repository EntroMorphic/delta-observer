#!/usr/bin/env python3
"""
Geometric analysis of 4-bit adder activation landscapes.

Use UMAP to project high-dimensional activations to 2D/3D and visualize
the geometric structure of how each model represents the problem.
"""

import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json


def load_activations(path):
    """Load activation landscapes."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


def concatenate_activations(activations, exclude_keys=['inputs', 'outputs']):
    """Concatenate all activation layers into single matrix."""
    keys = [k for k in sorted(activations.keys()) if k not in exclude_keys]
    matrices = [activations[k] for k in keys]
    concatenated = np.concatenate(matrices, axis=1)
    return concatenated, keys


def analyze_geometry(activations_path, model_name, output_dir):
    """
    Analyze geometric structure of activation landscape.
    
    Returns UMAP embeddings and analysis results.
    """
    print(f"\n{'='*80}")
    print(f"GEOMETRIC ANALYSIS: {model_name}")
    print(f"{'='*80}")
    
    # Load data
    print("\n📂 Loading activations...")
    data = load_activations(activations_path)
    inputs = data['inputs']
    outputs = data['outputs']
    
    # Concatenate all layers
    print("\n🔗 Concatenating activation layers...")
    X, layer_names = concatenate_activations(data)
    print(f"   Shape: {X.shape}")
    print(f"   Layers: {len(layer_names)}")
    
    # PCA for comparison
    print("\n📉 Running PCA...")
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    print(f"   First 10 components explain: {explained_var[:10].sum():.4f} of variance")
    
    # UMAP 2D
    print("\n🗺️  Running UMAP (2D)...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    embedding_2d = reducer_2d.fit_transform(X)
    print(f"   Embedding shape: {embedding_2d.shape}")
    
    # UMAP 3D
    print("\n🗺️  Running UMAP (3D)...")
    reducer_3d = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    embedding_3d = reducer_3d.fit_transform(X)
    print(f"   Embedding shape: {embedding_3d.shape}")
    
    # Decode inputs for coloring
    print("\n🎨 Decoding inputs for visualization...")
    a_vals = np.sum(inputs[:, 0:4] * np.array([1, 2, 4, 8]), axis=1).astype(int)
    b_vals = np.sum(inputs[:, 4:8] * np.array([1, 2, 4, 8]), axis=1).astype(int)
    carry_in = inputs[:, 8].astype(int)
    
    # Decode outputs
    sum_vals = np.sum(outputs[:, 0:4] * np.array([1, 2, 4, 8]), axis=1).astype(int)
    carry_out = outputs[:, 4].astype(int)
    
    # Save results
    results = {
        'model_name': model_name,
        'activation_shape': X.shape,
        'pca_variance_explained': explained_var.tolist(),
        'embedding_2d': embedding_2d.tolist(),
        'embedding_3d': embedding_3d.tolist(),
        'inputs': {
            'a': a_vals.tolist(),
            'b': b_vals.tolist(),
            'carry_in': carry_in.tolist(),
        },
        'outputs': {
            'sum': sum_vals.tolist(),
            'carry_out': carry_out.tolist(),
        },
    }
    
    with open(f"{output_dir}/{model_name}_geometry.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {model_name}_geometry.json")
    
    return results


def visualize_comparison(mono_results, comp_results, output_dir):
    """
    Create comparison visualizations of both models' geometric structures.
    """
    print(f"\n{'='*80}")
    print("VISUALIZATION: GEOMETRIC COMPARISON")
    print(f"{'='*80}")
    
    mono_emb = np.array(mono_results['embedding_2d'])
    comp_emb = np.array(comp_results['embedding_2d'])
    
    a_vals = np.array(mono_results['inputs']['a'])
    sum_vals = np.array(mono_results['outputs']['sum'])
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Monolithic - colored by input A
    ax = axes[0, 0]
    scatter = ax.scatter(mono_emb[:, 0], mono_emb[:, 1], c=a_vals, cmap='viridis', s=20, alpha=0.6)
    ax.set_title('Monolithic Model - Colored by Input A', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Input A (0-15)')
    ax.grid(True, alpha=0.3)
    
    # Monolithic - colored by output sum
    ax = axes[0, 1]
    scatter = ax.scatter(mono_emb[:, 0], mono_emb[:, 1], c=sum_vals, cmap='plasma', s=20, alpha=0.6)
    ax.set_title('Monolithic Model - Colored by Output Sum', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Output Sum (0-31)')
    ax.grid(True, alpha=0.3)
    
    # Compositional - colored by input A
    ax = axes[1, 0]
    scatter = ax.scatter(comp_emb[:, 0], comp_emb[:, 1], c=a_vals, cmap='viridis', s=20, alpha=0.6)
    ax.set_title('Compositional Model - Colored by Input A', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Input A (0-15)')
    ax.grid(True, alpha=0.3)
    
    # Compositional - colored by output sum
    ax = axes[1, 1]
    scatter = ax.scatter(comp_emb[:, 0], comp_emb[:, 1], c=sum_vals, cmap='plasma', s=20, alpha=0.6)
    ax.set_title('Compositional Model - Colored by Output Sum', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Output Sum (0-31)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/geometric_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved: geometric_comparison.png")
    plt.close()
    
    # Create individual high-res plots
    for model_name, emb in [('monolithic', mono_emb), ('compositional', comp_emb)]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # By input A
        ax = axes[0]
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=a_vals, cmap='viridis', s=30, alpha=0.7)
        ax.set_title(f'{model_name.capitalize()} - Input A Structure', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        plt.colorbar(scatter, ax=ax, label='Input A (0-15)')
        ax.grid(True, alpha=0.3)
        
        # By output sum
        ax = axes[1]
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=sum_vals, cmap='plasma', s=30, alpha=0.7)
        ax.set_title(f'{model_name.capitalize()} - Output Sum Structure', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        plt.colorbar(scatter, ax=ax, label='Output Sum (0-31)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_geometry_detailed.png", dpi=200, bbox_inches='tight')
        print(f"💾 Detailed visualization saved: {model_name}_geometry_detailed.png")
        plt.close()


def main():
    """Run geometric analysis on both models."""
    print("=" * 80)
    print("GEOMETRIC MICROSCOPE: 4-BIT ADDER ANALYSIS")
    print("=" * 80)
    
    output_dir = "/home/ubuntu/geometric-microscope/analysis"
    
    # Analyze monolithic model
    mono_results = analyze_geometry(
        "/home/ubuntu/geometric-microscope/analysis/monolithic_activations.npz",
        "monolithic",
        output_dir
    )
    
    # Analyze compositional model
    comp_results = analyze_geometry(
        "/home/ubuntu/geometric-microscope/analysis/compositional_activations.npz",
        "compositional",
        output_dir
    )
    
    # Create comparison visualizations
    visualize_comparison(mono_results, comp_results, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n✅ Geometric analysis complete!")
    print(f"\n📊 Results:")
    print(f"   - monolithic_geometry.json")
    print(f"   - compositional_geometry.json")
    print(f"   - geometric_comparison.png")
    print(f"   - monolithic_geometry_detailed.png")
    print(f"   - compositional_geometry_detailed.png")
    print(f"\n🔬 The shapes of computation are now visible!")


if __name__ == "__main__":
    main()
