#!/usr/bin/env python3
"""
Create a motion picture of the Delta Observer's training trajectory.
Watch the scaffolding rise and fall in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("Loading trajectory data...")

# Load trajectory data
trajectory = np.load('data/online_observer_trajectory.npz')
snapshots = trajectory['snapshots']  # (38, 512, 16)
epochs = trajectory['epochs']  # (38,)

# Load semantic labels (carry counts)
latents_data = np.load('data/online_observer_latents.npz')
carry_counts = latents_data['carry_counts']  # (512,) or similar

print(f"Loaded {len(epochs)} snapshots from epochs {epochs[0]} to {epochs[-1]}")
print(f"Each snapshot: {snapshots.shape[1]} samples in {snapshots.shape[2]}D")

# Compute global PCA on all snapshots combined for consistent projection
print("Computing consistent 2D projection...")
all_latents = snapshots.reshape(-1, 16)
pca = PCA(n_components=2)
pca.fit(all_latents)

# Project each snapshot
projections = []
for i in range(len(epochs)):
    proj = pca.transform(snapshots[i])
    projections.append(proj)

# Compute metrics for each snapshot
print("Computing metrics for each frame...")
r2_scores = []
silhouette_scores = []

for i in range(len(epochs)):
    z = snapshots[i]

    # R² (linear accessibility)
    reg = LinearRegression()
    reg.fit(z, carry_counts)
    r2 = reg.score(z, carry_counts)
    r2_scores.append(r2)

    # Silhouette (geometric clustering)
    try:
        sil = silhouette_score(z, carry_counts)
    except:
        sil = 0.0
    silhouette_scores.append(sil)

r2_scores = np.array(r2_scores)
silhouette_scores = np.array(silhouette_scores)

# Find global bounds for consistent axes
all_proj = np.vstack(projections)
x_min, x_max = all_proj[:, 0].min() - 0.5, all_proj[:, 0].max() + 0.5
y_min, y_max = all_proj[:, 1].min() - 0.5, all_proj[:, 1].max() + 0.5

# Create the animation
print("Creating animation...")
fig = plt.figure(figsize=(14, 8))

# Main latent space plot
ax_main = fig.add_axes([0.05, 0.15, 0.55, 0.75])

# Metrics plot (R² and Silhouette over time)
ax_metrics = fig.add_axes([0.65, 0.55, 0.30, 0.35])

# Info box
ax_info = fig.add_axes([0.65, 0.15, 0.30, 0.30])
ax_info.axis('off')

# Colors for carry counts
cmap = plt.cm.viridis
unique_carries = np.unique(carry_counts)
colors = cmap(np.linspace(0, 1, len(unique_carries)))
color_map = {c: colors[i] for i, c in enumerate(unique_carries)}
point_colors = [color_map[c] for c in carry_counts]

# Title
fig.suptitle('Delta Observer: Watching the Scaffolding Rise and Fall',
             fontsize=16, fontweight='bold', y=0.97)

def animate(frame_idx):
    ax_main.clear()
    ax_metrics.clear()
    ax_info.clear()
    ax_info.axis('off')

    epoch = epochs[frame_idx]
    proj = projections[frame_idx]
    r2 = r2_scores[frame_idx]
    sil = silhouette_scores[frame_idx]

    # Main scatter plot
    scatter = ax_main.scatter(proj[:, 0], proj[:, 1], c=carry_counts,
                              cmap='viridis', s=30, alpha=0.7, edgecolors='white', linewidth=0.3)

    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xlabel('PC1', fontsize=11)
    ax_main.set_ylabel('PC2', fontsize=11)
    ax_main.set_title(f'Latent Space at Epoch {epoch}', fontsize=13, fontweight='bold')

    # Add colorbar on first frame reference
    if frame_idx == 0:
        cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8, pad=0.02)
        cbar.set_label('Carry Count', fontsize=10)

    # Metrics plot
    ax_metrics.plot(epochs[:frame_idx+1], r2_scores[:frame_idx+1], 'g-', linewidth=2.5, label='R² (Accessibility)')
    ax_metrics.plot(epochs[:frame_idx+1], silhouette_scores[:frame_idx+1], 'r-', linewidth=2.5, label='Silhouette (Clustering)')

    # Current position markers
    ax_metrics.scatter([epoch], [r2], color='green', s=100, zorder=5, edgecolors='white', linewidth=2)
    ax_metrics.scatter([epoch], [sil], color='red', s=100, zorder=5, edgecolors='white', linewidth=2)

    ax_metrics.set_xlim(0, epochs[-1] + 5)
    ax_metrics.set_ylim(-0.1, 1.05)
    ax_metrics.set_xlabel('Epoch', fontsize=10)
    ax_metrics.set_ylabel('Score', fontsize=10)
    ax_metrics.set_title('Learning Dynamics', fontsize=12, fontweight='bold')
    ax_metrics.legend(loc='right', fontsize=8)
    ax_metrics.grid(True, alpha=0.3)

    # Highlight peak clustering region
    if 15 <= epoch <= 30:
        ax_metrics.axvspan(15, 30, alpha=0.2, color='orange', label='Peak Scaffolding')

    # Info box with current metrics
    phase = "Initialization" if epoch < 5 else \
            "Early Learning" if epoch < 15 else \
            "Peak Scaffolding" if epoch < 35 else \
            "Dissolving" if epoch < 60 else \
            "Converged"

    info_text = f"""
    Epoch: {epoch}

    R² (Accessibility): {r2:.4f}
    Silhouette (Clustering): {sil:.4f}

    Phase: {phase}
    """

    # Color the phase text
    phase_colors = {
        "Initialization": "#666666",
        "Early Learning": "#2196F3",
        "Peak Scaffolding": "#FF9800",
        "Dissolving": "#9C27B0",
        "Converged": "#4CAF50"
    }

    ax_info.text(0.5, 0.5, info_text.strip(), transform=ax_info.transAxes,
                 fontsize=11, verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 family='monospace')

    return scatter,

# Create animation
print(f"Rendering {len(epochs)} frames...")
anim = FuncAnimation(fig, animate, frames=len(epochs), interval=200, blit=False)

# Save as MP4
output_path = 'figures/delta_observer_training.mp4'
print(f"Saving to {output_path}...")

writer = FFMpegWriter(fps=5, metadata=dict(artist='Delta Observer'), bitrate=2000)
anim.save(output_path, writer=writer, dpi=120)

print(f"\nDone! Movie saved to {output_path}")
print(f"Duration: {len(epochs) / 5:.1f} seconds at 5 fps")

# Also save a GIF version for easy sharing
gif_path = 'figures/delta_observer_training.gif'
print(f"Saving GIF to {gif_path}...")
anim.save(gif_path, writer='pillow', fps=5, dpi=80)
print("GIF saved!")

plt.close()
