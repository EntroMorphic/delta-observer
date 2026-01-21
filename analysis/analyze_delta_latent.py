#!/usr/bin/env python3
"""
Analyze Delta Observer latent space for emergent semantic structure.

Key questions:
1. Does the latent space cluster by carry_count? (semantic structure)
2. Does the latent space cluster by bit_position? (compositional structure)
3. Are there interpretable directions in latent space?
4. Does structure generalize to held-out samples?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import umap
import sys
sys.path.append('/home/ubuntu/geometric-microscope')

from models.delta_observer import DeltaObserver, DeltaObserverDataset


def load_model_and_data():
    """Load trained Delta Observer and dataset."""
    print("Loading model and data...")
    
    # Load model
    model = DeltaObserver(mono_dim=64, comp_dim=64, latent_dim=16)
    checkpoint = torch.load('/home/ubuntu/geometric-microscope/models/delta_observer_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset = DeltaObserverDataset('/home/ubuntu/geometric-microscope/analysis/delta_observer_dataset.npz')
    
    print("✅ Model and data loaded")
    return model, dataset


def extract_latent_representations(model, dataset):
    """Extract latent representations for all samples."""
    print("\n🔬 Extracting latent representations...")
    
    latents = []
    carry_counts = []
    bit_positions = []
    inputs = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            mono_act = sample['mono_act'].unsqueeze(0)
            comp_act = sample['comp_act'].unsqueeze(0)
            
            latent = model.encode(mono_act, comp_act)
            latents.append(latent.squeeze(0).numpy())
            carry_counts.append(sample['carry_count'].item())
            bit_positions.append(sample['bit_position'].item())
            inputs.append(sample['input'].numpy())
    
    latents = np.array(latents)
    carry_counts = np.array(carry_counts)
    bit_positions = np.array(bit_positions)
    inputs = np.array(inputs)
    
    print(f"   Latent shape: {latents.shape}")
    print(f"   Carry count range: {carry_counts.min()}-{carry_counts.max()}")
    print(f"   Bit position range: {bit_positions.min()}-{bit_positions.max()}")
    
    return latents, carry_counts, bit_positions, inputs


def analyze_clustering(latents, carry_counts, bit_positions):
    """Analyze clustering structure in latent space."""
    print("\n📊 Analyzing clustering structure...")
    
    # Silhouette score by carry_count
    if len(np.unique(carry_counts)) > 1:
        sil_carry = silhouette_score(latents, carry_counts)
        print(f"   Silhouette score (carry_count): {sil_carry:.4f}")
    
    # Silhouette score by bit_position
    if len(np.unique(bit_positions)) > 1:
        sil_bit = silhouette_score(latents, bit_positions)
        print(f"   Silhouette score (bit_position): {sil_bit:.4f}")
    
    return sil_carry, sil_bit


def analyze_linear_probes(latents, carry_counts, bit_positions):
    """Train linear probes to test if semantic info is linearly accessible."""
    print("\n🎯 Training linear probes...")
    
    # Probe for carry_count
    reg_carry = LinearRegression()
    reg_carry.fit(latents, carry_counts)
    carry_pred = reg_carry.predict(latents)
    carry_r2 = reg_carry.score(latents, carry_counts)
    carry_corr, _ = pearsonr(carry_counts, carry_pred)
    
    print(f"   Carry count prediction:")
    print(f"     R² score: {carry_r2:.4f}")
    print(f"     Correlation: {carry_corr:.4f}")
    
    # Probe for bit_position
    reg_bit = LinearRegression()
    reg_bit.fit(latents, bit_positions)
    bit_pred = reg_bit.predict(latents)
    bit_r2 = reg_bit.score(latents, bit_positions)
    bit_corr, _ = pearsonr(bit_positions, bit_pred)
    
    print(f"   Bit position prediction:")
    print(f"     R² score: {bit_r2:.4f}")
    print(f"     Correlation: {bit_corr:.4f}")
    
    return {
        'carry_r2': carry_r2,
        'carry_corr': carry_corr,
        'bit_r2': bit_r2,
        'bit_corr': bit_corr,
    }


def visualize_latent_space(latents, carry_counts, bit_positions, inputs):
    """Create comprehensive visualizations of latent space."""
    print("\n🎨 Creating visualizations...")
    
    # 1. UMAP 2D projection
    print("   Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    latent_2d = reducer.fit_transform(latents)
    
    # 2. PCA for variance analysis
    print("   Computing PCA...")
    pca = PCA()
    pca.fit(latents)
    explained_var = pca.explained_variance_ratio_
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: UMAP colored by carry_count
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                          c=carry_counts, cmap='viridis', 
                          s=20, alpha=0.6)
    ax1.set_title('Latent Space (UMAP) - Colored by Carry Count', fontsize=14, fontweight='bold')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    plt.colorbar(scatter1, ax=ax1, label='Carry Count')
    
    # Plot 2: UMAP colored by bit_position
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                          c=bit_positions, cmap='tab10', 
                          s=20, alpha=0.6)
    ax2.set_title('Latent Space (UMAP) - Colored by Bit Position', fontsize=14, fontweight='bold')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    plt.colorbar(scatter2, ax=ax2, label='Bit Position')
    
    # Plot 3: UMAP colored by input sum
    ax3 = plt.subplot(2, 3, 3)
    input_sums = inputs[:, :8].sum(axis=1)  # Sum of a and b bits
    scatter3 = ax3.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                          c=input_sums, cmap='coolwarm', 
                          s=20, alpha=0.6)
    ax3.set_title('Latent Space (UMAP) - Colored by Input Sum', fontsize=14, fontweight='bold')
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    plt.colorbar(scatter3, ax=ax3, label='Input Sum')
    
    # Plot 4: PCA variance explained
    ax4 = plt.subplot(2, 3, 4)
    ax4.bar(range(1, len(explained_var)+1), explained_var)
    ax4.set_title('PCA Variance Explained', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Variance Explained')
    ax4.set_xticks(range(1, len(explained_var)+1))
    
    # Plot 5: Carry count distribution per cluster
    ax5 = plt.subplot(2, 3, 5)
    carry_counts_per_bit = [carry_counts[bit_positions == i] for i in range(4)]
    ax5.boxplot(carry_counts_per_bit, labels=['Bit 0', 'Bit 1', 'Bit 2', 'Bit 3'])
    ax5.set_title('Carry Count Distribution by Bit Position', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Bit Position')
    ax5.set_ylabel('Carry Count')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Latent dimension heatmap (first 100 samples)
    ax6 = plt.subplot(2, 3, 6)
    im = ax6.imshow(latents[:100].T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax6.set_title('Latent Activations (First 100 Samples)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('Latent Dimension')
    plt.colorbar(im, ax=ax6, label='Activation')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/geometric-microscope/analysis/delta_latent_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("   ✅ Saved: delta_latent_analysis.png")
    
    # Save UMAP coordinates for further analysis
    np.savez('/home/ubuntu/geometric-microscope/analysis/delta_latent_umap.npz',
             latent_2d=latent_2d,
             latents=latents,
             carry_counts=carry_counts,
             bit_positions=bit_positions)
    print("   ✅ Saved: delta_latent_umap.npz")
    
    return latent_2d, explained_var


def test_perturbation_stability(model, dataset, n_samples=20):
    """Test if latent space is stable under input perturbations."""
    print("\n🔬 Testing perturbation stability...")
    
    perturbation_distances = []
    
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            mono_act = sample['mono_act'].unsqueeze(0)
            comp_act = sample['comp_act'].unsqueeze(0)
            
            # Original latent
            latent_orig = model.encode(mono_act, comp_act).numpy()
            
            # Perturb activations slightly
            mono_act_perturbed = mono_act + torch.randn_like(mono_act) * 0.1
            comp_act_perturbed = comp_act + torch.randn_like(comp_act) * 0.1
            
            # Perturbed latent
            latent_perturbed = model.encode(mono_act_perturbed, comp_act_perturbed).numpy()
            
            # Measure distance
            dist = np.linalg.norm(latent_orig - latent_perturbed)
            perturbation_distances.append(dist)
    
    mean_dist = np.mean(perturbation_distances)
    std_dist = np.std(perturbation_distances)
    
    print(f"   Mean perturbation distance: {mean_dist:.4f} ± {std_dist:.4f}")
    print(f"   (Lower = more stable, meaning is robust)")
    
    return mean_dist, std_dist


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("DELTA OBSERVER LATENT SPACE ANALYSIS")
    print("="*80)
    
    # Load
    model, dataset = load_model_and_data()
    
    # Extract latents
    latents, carry_counts, bit_positions, inputs = extract_latent_representations(model, dataset)
    
    # Analyze clustering
    sil_carry, sil_bit = analyze_clustering(latents, carry_counts, bit_positions)
    
    # Linear probes
    probe_results = analyze_linear_probes(latents, carry_counts, bit_positions)
    
    # Visualize
    latent_2d, explained_var = visualize_latent_space(latents, carry_counts, bit_positions, inputs)
    
    # Perturbation stability
    mean_dist, std_dist = test_perturbation_stability(model, dataset)
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print("\n📊 Clustering Quality:")
    print(f"   Silhouette (carry_count): {sil_carry:.4f}")
    print(f"   Silhouette (bit_position): {sil_bit:.4f}")
    
    print("\n🎯 Linear Probe Results:")
    print(f"   Carry count R²: {probe_results['carry_r2']:.4f}")
    print(f"   Bit position R²: {probe_results['bit_r2']:.4f}")
    
    print("\n📈 PCA Variance:")
    print(f"   Top 3 components: {explained_var[:3].sum():.2%}")
    print(f"   Top 5 components: {explained_var[:5].sum():.2%}")
    
    print("\n🔬 Perturbation Stability:")
    print(f"   Mean distance: {mean_dist:.4f}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if sil_carry > 0.5:
        print("\n✅ STRONG SEMANTIC STRUCTURE DETECTED")
        print("   The latent space clusters by carry_count (silhouette > 0.5)")
        print("   This suggests the Delta Observer learned semantic meaning!")
    elif sil_carry > 0.3:
        print("\n⚠️  MODERATE SEMANTIC STRUCTURE")
        print("   The latent space shows some carry_count structure")
        print("   But it's not as clear as we'd hope")
    else:
        print("\n❌ WEAK SEMANTIC STRUCTURE")
        print("   The latent space does not clearly cluster by carry_count")
        print("   This suggests translation without deep meaning")
    
    if probe_results['carry_r2'] > 0.7:
        print("\n✅ CARRY COUNT IS LINEARLY ACCESSIBLE")
        print(f"   Linear probe achieves R² = {probe_results['carry_r2']:.4f}")
        print("   Semantic information is explicitly represented!")
    
    if mean_dist < 1.0:
        print("\n✅ LATENT SPACE IS STABLE")
        print("   Small perturbations cause small latent changes")
        print("   This suggests robust semantic representation")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\n📁 Outputs saved:")
    print("   - analysis/delta_latent_analysis.png")
    print("   - analysis/delta_latent_umap.npz")


if __name__ == "__main__":
    main()
