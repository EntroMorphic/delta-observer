#!/usr/bin/env python3
"""
Generate figures for the Online Delta Observer paper.

Creates figures showing:
1. Transient clustering evolution (Silhouette over epochs)
2. R² and Silhouette trajectory (dual axis)
3. Latent space visualization (UMAP colored by carry count)
4. Method comparison bar chart
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, r2_score
from pathlib import Path

# Try to import UMAP, fall back to PCA if not available
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    from sklearn.decomposition import PCA
    HAS_UMAP = False
    print("UMAP not available, using PCA for visualization")

# Setup paths
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style settings
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12

# Colors
COLORS = {
    'r2': '#2ecc71',        # Green
    'silhouette': '#e74c3c', # Red
    'online': '#3498db',     # Blue
    'posthoc': '#9b59b6',    # Purple
    'pca': '#95a5a6',        # Gray
    'carry': plt.cm.viridis  # Colormap for carry counts
}


def load_data():
    """Load online observer data."""
    trajectory = np.load(DATA_DIR / "online_observer_trajectory.npz")
    latents = np.load(DATA_DIR / "online_observer_latents.npz")
    return trajectory, latents


def compute_trajectory_metrics(trajectory, latents):
    """Compute R² and Silhouette at each epoch in the trajectory."""
    snapshots = trajectory['snapshots']  # (n_epochs, n_samples, n_dims)
    epochs = trajectory['epochs']
    carry_counts = latents['carry_counts']

    r2_values = []
    silhouette_values = []

    for i, epoch in enumerate(epochs):
        z = snapshots[i]  # (n_samples, n_dims)

        # R² via linear regression
        reg = LinearRegression()
        reg.fit(z, carry_counts)
        pred = reg.predict(z)
        r2 = r2_score(carry_counts, pred)
        r2_values.append(r2)

        # Silhouette score
        # Need at least 2 clusters with >1 sample each
        unique_counts = np.unique(carry_counts)
        if len(unique_counts) >= 2:
            try:
                sil = silhouette_score(z, carry_counts)
            except:
                sil = 0.0
        else:
            sil = 0.0
        silhouette_values.append(sil)

    return epochs, np.array(r2_values), np.array(silhouette_values)


def figure1_transient_clustering(epochs, r2_values, silhouette_values):
    """
    Figure: R² and Silhouette evolution during training.
    Shows the transient clustering phenomenon.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # R² on left axis
    color1 = COLORS['r2']
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('R² (Linear Accessibility)', color=color1)
    line1, = ax1.plot(epochs, r2_values, color=color1, linewidth=2.5,
                      marker='o', markersize=4, label='R² (Accessibility)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.9, color=color1, linestyle='--', alpha=0.3, linewidth=1)

    # Silhouette on right axis
    ax2 = ax1.twinx()
    color2 = COLORS['silhouette']
    ax2.set_ylabel('Silhouette Score (Clustering)', color=color2)
    line2, = ax2.plot(epochs, silhouette_values, color=color2, linewidth=2.5,
                      marker='s', markersize=4, label='Silhouette (Clustering)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.1, 0.5)
    ax2.axhline(y=0, color=color2, linestyle='--', alpha=0.3, linewidth=1)

    # Find and annotate peak clustering
    peak_idx = np.argmax(silhouette_values)
    peak_epoch = epochs[peak_idx]
    peak_sil = silhouette_values[peak_idx]
    ax2.annotate(f'Peak: {peak_sil:.2f}\n(epoch {peak_epoch})',
                 xy=(peak_epoch, peak_sil),
                 xytext=(peak_epoch + 20, peak_sil + 0.08),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color=color2, alpha=0.7),
                 color=color2)

    # Title and legend
    ax1.set_title('Transient Clustering: Geometric Structure Emerges Then Dissolves',
                  fontsize=14, fontweight='bold')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)

    # Add annotation explaining the phenomenon
    fig.text(0.5, 0.02,
             'Clustering peaks during learning then dissolves—scaffolding, not structure.',
             ha='center', fontsize=10, style='italic', color='#555555')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    save_path = FIGURES_DIR / "figure5_training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def figure2_latent_space(latents):
    """
    Figure: UMAP visualization of latent space colored by carry count.
    """
    z = latents['latents']
    carry_counts = latents['carry_counts']

    # Dimensionality reduction
    if HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        z_2d = reducer.fit_transform(z)
        method_name = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=42)
        z_2d = reducer.fit_transform(z)
        method_name = "PCA"

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1],
                         c=carry_counts, cmap='viridis',
                         s=50, alpha=0.7, edgecolors='white', linewidth=0.5)

    cbar = plt.colorbar(scatter, ax=ax, label='Carry Count')
    cbar.set_ticks([0, 1, 2, 3, 4])

    ax.set_xlabel(f'{method_name} Dimension 1')
    ax.set_ylabel(f'{method_name} Dimension 2')
    ax.set_title('Online Delta Observer Latent Space\n(Colored by Carry Count)',
                 fontsize=14, fontweight='bold')

    # Add R² and Silhouette annotations
    reg = LinearRegression().fit(z, carry_counts)
    r2 = r2_score(carry_counts, reg.predict(z))
    sil = silhouette_score(z, carry_counts)

    textstr = f'R² = {r2:.4f}\nSilhouette = {sil:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    save_path = FIGURES_DIR / "figure2_delta_latent_space.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def figure3_accessibility_vs_clustering(latents):
    """
    Figure: Scatter plot showing accessibility vs clustering for different methods.
    """
    z = latents['latents']
    carry_counts = latents['carry_counts']
    mono_act = latents['mono_activations']
    comp_act = latents['comp_activations']

    # Compute metrics for each representation
    methods = []

    # Online observer
    reg = LinearRegression().fit(z, carry_counts)
    r2_online = r2_score(carry_counts, reg.predict(z))
    sil_online = silhouette_score(z, carry_counts)
    methods.append(('Online Observer', r2_online, sil_online, COLORS['online']))

    # PCA on combined activations
    from sklearn.decomposition import PCA
    combined = np.concatenate([mono_act, comp_act], axis=1)
    pca = PCA(n_components=16, random_state=42)
    z_pca = pca.fit_transform(combined)
    reg = LinearRegression().fit(z_pca, carry_counts)
    r2_pca = r2_score(carry_counts, reg.predict(z_pca))
    sil_pca = silhouette_score(z_pca, carry_counts)
    methods.append(('PCA Baseline', r2_pca, sil_pca, COLORS['pca']))

    # PCA on compositional only
    pca_comp = PCA(n_components=16, random_state=42)
    z_pca_comp = pca_comp.fit_transform(comp_act)
    reg = LinearRegression().fit(z_pca_comp, carry_counts)
    r2_pca_comp = r2_score(carry_counts, reg.predict(z_pca_comp))
    sil_pca_comp = silhouette_score(z_pca_comp, carry_counts)
    methods.append(('PCA (Compositional)', r2_pca_comp, sil_pca_comp, '#f39c12'))

    fig, ax = plt.subplots(figsize=(10, 7))

    for name, r2, sil, color in methods:
        ax.scatter(sil, r2, s=200, c=color, label=name, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(name, (sil, r2), xytext=(10, 5), textcoords='offset points', fontsize=10)

    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='High Accessibility (R²>0.95)')
    ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Low Clustering (Sil<0.1)')

    # Highlight the "good" region
    ax.fill_between([-0.1, 0.1], 0.95, 1.0, alpha=0.1, color='green')
    ax.text(0.0, 0.97, 'High Accessibility\nLow Clustering', ha='center', fontsize=9, color='green')

    ax.set_xlabel('Silhouette Score (Geometric Clustering)', fontsize=12)
    ax.set_ylabel('R² (Linear Accessibility)', fontsize=12)
    ax.set_title('Accessibility vs Clustering: The Dissociation', fontsize=14, fontweight='bold')

    ax.set_xlim(-0.15, 0.6)
    ax.set_ylim(0.7, 1.02)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = FIGURES_DIR / "figure3_accessibility_vs_clustering.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def figure4_method_comparison():
    """
    Figure: Bar chart comparing Online vs Post-hoc vs PCA.
    """
    methods = ['Online\nObserver', 'Post-hoc\nObserver', 'PCA\nBaseline']
    r2_values = [0.9879, 0.9505, 0.9482]
    colors = [COLORS['online'], COLORS['posthoc'], COLORS['pca']]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(methods, r2_values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add delta annotations
    ax.annotate('', xy=(0, 0.9879), xytext=(2, 0.9482),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(1, 0.965, '+4.0%', ha='center', fontsize=11, fontweight='bold', color='green')

    ax.set_ylabel('R² (Linear Accessibility)', fontsize=12)
    ax.set_title('Method Comparison: Online Observation Wins', fontsize=14, fontweight='bold')
    ax.set_ylim(0.93, 1.01)
    ax.axhline(y=0.9482, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    save_path = FIGURES_DIR / "figure_method_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def figure5_trajectory_phases(epochs, r2_values, silhouette_values):
    """
    Figure: Annotated trajectory showing the three phases.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot both metrics normalized to [0,1] for comparison
    r2_norm = r2_values
    sil_norm = (silhouette_values - silhouette_values.min()) / (silhouette_values.max() - silhouette_values.min() + 1e-8)

    ax.fill_between(epochs, 0, 1, where=(epochs <= 10), alpha=0.2, color='blue', label='Phase 1: Init')
    ax.fill_between(epochs, 0, 1, where=((epochs > 10) & (epochs <= 50)), alpha=0.2, color='green', label='Phase 2: Learning')
    ax.fill_between(epochs, 0, 1, where=(epochs > 50), alpha=0.2, color='orange', label='Phase 3: Convergence')

    ax.plot(epochs, r2_values, color=COLORS['r2'], linewidth=2.5, marker='o', markersize=4, label='R²')
    ax.plot(epochs, silhouette_values, color=COLORS['silhouette'], linewidth=2.5, marker='s', markersize=4, label='Silhouette')

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Phase labels
    ax.text(5, 0.85, 'Init\n(Random)', ha='center', fontsize=10, color='blue')
    ax.text(30, 0.85, 'Learning\n(Scaffolding)', ha='center', fontsize=10, color='green')
    ax.text(125, 0.85, 'Converged\n(Scaffolding Gone)', ha='center', fontsize=10, color='orange')

    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Three Phases of Representation Learning', fontsize=14, fontweight='bold')
    ax.legend(loc='center right', fontsize=10)
    ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()

    save_path = FIGURES_DIR / "figure_trajectory_phases.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("Loading data...")
    trajectory, latents = load_data()

    print("Computing trajectory metrics...")
    epochs, r2_values, silhouette_values = compute_trajectory_metrics(trajectory, latents)

    print(f"\nTrajectory Summary:")
    print(f"  Epochs: {epochs[0]} to {epochs[-1]} ({len(epochs)} snapshots)")
    print(f"  R² range: {r2_values.min():.4f} to {r2_values.max():.4f}")
    print(f"  Silhouette range: {silhouette_values.min():.4f} to {silhouette_values.max():.4f}")

    peak_idx = np.argmax(silhouette_values)
    print(f"  Peak clustering: Silhouette={silhouette_values[peak_idx]:.4f} at epoch {epochs[peak_idx]}")

    print("\nGenerating figures...")

    figure1_transient_clustering(epochs, r2_values, silhouette_values)
    figure2_latent_space(latents)
    figure3_accessibility_vs_clustering(latents)
    figure4_method_comparison()
    figure5_trajectory_phases(epochs, r2_values, silhouette_values)

    print("\nAll figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
