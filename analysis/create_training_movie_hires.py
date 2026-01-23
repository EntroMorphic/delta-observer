#!/usr/bin/env python3
"""
High-Resolution Training Movie
==============================
Every epoch, every frame. Watch the scaffolding rise and fall in full detail.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HIGH-RESOLUTION TRAINING MOVIE")
print("=" * 70)

# Load high-res trajectory
print("\nLoading high-resolution trajectory...")
data = np.load('data/online_observer_trajectory_hires.npz')
snapshots = data['snapshots']  # (200, 512, 16)
epochs = data['epochs']  # (200,)
carry_counts = data['carry_counts']  # (512,)

print(f"Loaded {len(epochs)} frames (epochs 0-{epochs[-1]})")
print(f"Each frame: {snapshots.shape[1]} samples in {snapshots.shape[2]}D")

# Compute global PCA for consistent projection
print("\nComputing consistent 2D projection across all frames...")
all_latents = snapshots.reshape(-1, 16)
pca = PCA(n_components=2)
pca.fit(all_latents)

# Project each snapshot
projections = []
for i in range(len(epochs)):
    proj = pca.transform(snapshots[i])
    projections.append(proj)

# Compute metrics for each frame
print("Computing R² and Silhouette for each frame...")
r2_scores = []
silhouette_scores = []

for i in range(len(epochs)):
    z = snapshots[i]

    # R²
    reg = LinearRegression()
    reg.fit(z, carry_counts)
    r2 = reg.score(z, carry_counts)
    r2_scores.append(r2)

    # Silhouette
    try:
        sil = silhouette_score(z, carry_counts)
    except:
        sil = 0.0
    silhouette_scores.append(sil)

r2_scores = np.array(r2_scores)
silhouette_scores = np.array(silhouette_scores)

# Find peak scaffolding
peak_idx = np.argmax(silhouette_scores)
peak_epoch = epochs[peak_idx]
peak_sil = silhouette_scores[peak_idx]
print(f"\nPeak scaffolding at epoch {peak_epoch} (Silhouette = {peak_sil:.4f})")

# Global bounds
all_proj = np.vstack(projections)
x_min, x_max = all_proj[:, 0].min() - 0.5, all_proj[:, 0].max() + 0.5
y_min, y_max = all_proj[:, 1].min() - 0.5, all_proj[:, 1].max() + 0.5

# Create the animation
print("\nCreating high-resolution animation...")
fig = plt.figure(figsize=(16, 9))

# Main latent space plot - leave room for colorbar
ax_main = fig.add_axes([0.05, 0.12, 0.50, 0.78])

# Colorbar axis (fixed position)
ax_cbar = fig.add_axes([0.56, 0.12, 0.02, 0.78])

# Metrics plot
ax_metrics = fig.add_axes([0.65, 0.52, 0.32, 0.38])

# Info box
ax_info = fig.add_axes([0.65, 0.12, 0.32, 0.35])
ax_info.axis('off')

# Title
fig.suptitle('Delta Observer: The Scaffolding Rises and Falls',
             fontsize=18, fontweight='bold', y=0.97)

# Create colorbar once with a dummy mappable
import matplotlib.cm as cm
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=carry_counts.min(), vmax=carry_counts.max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax_cbar)
cbar.set_label('Carry Count', fontsize=11)

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
                              cmap='viridis', s=25, alpha=0.8,
                              edgecolors='white', linewidth=0.2,
                              vmin=carry_counts.min(), vmax=carry_counts.max())

    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xlabel('PC1', fontsize=12)
    ax_main.set_ylabel('PC2', fontsize=12)
    ax_main.set_title(f'Latent Space — Epoch {epoch}', fontsize=14, fontweight='bold')

    # === Metrics Plot ===
    # Full trajectory in gray
    ax_metrics.plot(epochs, r2_scores, 'g-', linewidth=1, alpha=0.3)
    ax_metrics.plot(epochs, silhouette_scores, 'r-', linewidth=1, alpha=0.3)

    # Traced trajectory up to current frame
    ax_metrics.plot(epochs[:frame_idx+1], r2_scores[:frame_idx+1],
                    'g-', linewidth=2.5, label='R² (Accessibility)')
    ax_metrics.plot(epochs[:frame_idx+1], silhouette_scores[:frame_idx+1],
                    'r-', linewidth=2.5, label='Silhouette (Clustering)')

    # Current position markers
    ax_metrics.scatter([epoch], [r2], color='#2E7D32', s=120, zorder=5,
                       edgecolors='white', linewidth=2)
    ax_metrics.scatter([epoch], [sil], color='#C62828', s=120, zorder=5,
                       edgecolors='white', linewidth=2)

    # Mark peak scaffolding
    ax_metrics.axvline(x=peak_epoch, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_metrics.annotate(f'Peak\n(epoch {peak_epoch})', xy=(peak_epoch, peak_sil),
                        xytext=(peak_epoch + 15, peak_sil + 0.15),
                        fontsize=8, color='orange', alpha=0.8)

    ax_metrics.set_xlim(-5, epochs[-1] + 5)
    ax_metrics.set_ylim(-0.15, 1.1)
    ax_metrics.set_xlabel('Epoch', fontsize=11)
    ax_metrics.set_ylabel('Score', fontsize=11)
    ax_metrics.set_title('Learning Dynamics', fontsize=13, fontweight='bold')
    ax_metrics.legend(loc='upper right', fontsize=9)
    ax_metrics.grid(True, alpha=0.3)

    # === Info Box ===
    # Determine phase
    if epoch < 5:
        phase = "🌱 Initialization"
        phase_color = "#9E9E9E"
    elif epoch < peak_epoch - 5:
        phase = "📈 Scaffolding Rising"
        phase_color = "#FF9800"
    elif epoch < peak_epoch + 10:
        phase = "🏗️ Peak Scaffolding"
        phase_color = "#F44336"
    elif epoch < 80:
        phase = "📉 Scaffolding Dissolving"
        phase_color = "#9C27B0"
    else:
        phase = "✅ Converged"
        phase_color = "#4CAF50"

    info_text = f"""Epoch: {epoch}

R² (Accessibility):     {r2:.4f}
Silhouette (Clustering): {sil:.4f}

Phase: {phase}

Peak clustering was at epoch {peak_epoch}
with Silhouette = {peak_sil:.4f}"""

    ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                 fontsize=11, verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9),
                 family='monospace')

    return scatter,

# Progress bar axis (created once, outside animate)
ax_progress = fig.add_axes([0.05, 0.02, 0.9, 0.03])

def animate_with_progress(frame_idx):
    result = animate(frame_idx)

    # Update progress bar
    ax_progress.clear()
    progress = (frame_idx + 1) / len(epochs)
    ax_progress.barh(0, 1, height=1, color='#E0E0E0', alpha=0.3)
    ax_progress.barh(0, progress, height=1, color='#2196F3', alpha=0.8)
    ax_progress.set_xlim(0, 1)
    ax_progress.set_ylim(-0.5, 0.5)
    ax_progress.axis('off')
    ax_progress.text(0.5, 0, f'Frame {frame_idx + 1}/{len(epochs)}',
                     ha='center', va='center', fontsize=9, color='#333')

    return result


# Create animation
print(f"Rendering {len(epochs)} frames...")
anim = FuncAnimation(fig, animate_with_progress, frames=len(epochs), interval=50, blit=False)

# Save as high-quality MP4
output_mp4 = 'figures/delta_observer_training_hires.mp4'
print(f"\nSaving MP4 to {output_mp4}...")
writer = FFMpegWriter(fps=20, metadata=dict(artist='Delta Observer'), bitrate=4000)
anim.save(output_mp4, writer=writer, dpi=150)
print(f"Saved! Duration: {len(epochs) / 20:.1f} seconds at 20 fps")

# Also save a GIF (lower quality for sharing)
output_gif = 'figures/delta_observer_training_hires.gif'
print(f"\nSaving GIF to {output_gif}...")
anim.save(output_gif, writer='pillow', fps=15, dpi=80)
print("GIF saved!")

plt.close()

print("\n" + "=" * 70)
print("DONE!")
print(f"  MP4: {output_mp4} (high quality, 20 fps)")
print(f"  GIF: {output_gif} (shareable, 15 fps)")
print("=" * 70)
