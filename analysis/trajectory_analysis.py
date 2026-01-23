#!/usr/bin/env python3
"""
Trajectory Analysis
===================
Analyze what the online observer learned from the training trajectory.

Key questions:
1. How did the latent space evolve during training?
2. When did the R² jump? (early epochs vs late)
3. Is there structure in the trajectory that post-hoc misses?
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
import json


def analyze_trajectory():
    print("=" * 80)
    print("TRAJECTORY ANALYSIS")
    print("=" * 80)

    # Load trajectory
    traj = np.load('data/online_observer_trajectory.npz', allow_pickle=True)
    snapshots = traj['snapshots']
    epochs = traj['epochs']

    # Load labels
    data = np.load('data/online_observer_latents.npz')
    carry_counts = data['carry_counts']

    print(f"\nTrajectory: {len(snapshots)} snapshots at epochs {list(epochs)}")

    # Analyze R² at each snapshot
    print("\n" + "-" * 60)
    print("R² Evolution During Training")
    print("-" * 60)
    print(f"{'Epoch':>8} {'R²':>12} {'Silhouette':>12} {'Delta R²':>12}")
    print("-" * 60)

    r2_history = []
    sil_history = []
    prev_r2 = 0

    for i, (epoch, latent) in enumerate(zip(epochs, snapshots)):
        reg = LinearRegression()
        reg.fit(latent, carry_counts)
        r2 = reg.score(latent, carry_counts)

        try:
            sil = silhouette_score(latent, carry_counts)
        except:
            sil = np.nan

        delta = r2 - prev_r2
        print(f"{epoch:>8} {r2:>12.4f} {sil:>12.4f} {delta:>+12.4f}")

        r2_history.append(r2)
        sil_history.append(sil)
        prev_r2 = r2

    # Find when 90% of final R² was achieved
    final_r2 = r2_history[-1]
    threshold_90 = 0.9 * final_r2

    for i, (epoch, r2) in enumerate(zip(epochs, r2_history)):
        if r2 >= threshold_90:
            print(f"\n90% of final R² achieved at epoch {epoch}")
            break

    # Analyze latent space drift
    print("\n" + "-" * 60)
    print("Latent Space Drift Analysis")
    print("-" * 60)

    if len(snapshots) > 1:
        drifts = []
        for i in range(1, len(snapshots)):
            drift = np.linalg.norm(snapshots[i] - snapshots[i-1], axis=1).mean()
            drifts.append(drift)
            print(f"Epoch {epochs[i-1]} → {epochs[i]}: mean drift = {drift:.4f}")

        # When did drift stabilize?
        drift_threshold = 0.1 * max(drifts)
        for i, drift in enumerate(drifts):
            if drift < drift_threshold:
                print(f"\nDrift stabilized (< 10% of max) after epoch {epochs[i]}")
                break

    # Compare early vs late snapshots
    print("\n" + "-" * 60)
    print("Early vs Late Latent Structure")
    print("-" * 60)

    if len(snapshots) >= 3:
        early_latent = snapshots[1]  # Epoch ~10
        late_latent = snapshots[-1]  # Final

        # Correlation between early and late latents
        corrs = []
        for j in range(early_latent.shape[1]):
            r, _ = pearsonr(early_latent[:, j], late_latent[:, j])
            corrs.append(r)

        print(f"Mean correlation between early (epoch {epochs[1]}) and late (epoch {epochs[-1]}) latents:")
        print(f"  Per-dimension: {np.mean(corrs):.4f} ± {np.std(corrs):.4f}")

        # Did the observer learn something from early epochs that it kept?
        early_reg = LinearRegression()
        early_reg.fit(early_latent, carry_counts)
        early_r2 = early_reg.score(early_latent, carry_counts)

        print(f"\nEarly epoch R²: {early_r2:.4f}")
        print(f"Final epoch R²: {r2_history[-1]:.4f}")
        print(f"Improvement: +{r2_history[-1] - early_r2:.4f}")

    # Temporal structure test: can we predict epoch from latent?
    print("\n" + "-" * 60)
    print("Temporal Structure Test")
    print("-" * 60)

    # Create dataset: (latent, epoch) for each snapshot
    all_latents = []
    all_epochs = []
    for epoch, latent in zip(epochs, snapshots):
        for sample_latent in latent:
            all_latents.append(sample_latent)
            all_epochs.append(epoch)

    all_latents = np.array(all_latents)
    all_epochs = np.array(all_epochs)

    # Can we predict epoch from latent?
    epoch_reg = LinearRegression()
    epoch_reg.fit(all_latents, all_epochs)
    epoch_r2 = epoch_reg.score(all_latents, all_epochs)

    print(f"Predicting epoch from latent: R² = {epoch_r2:.4f}")

    if epoch_r2 > 0.5:
        print("  → The latent space encodes temporal information!")
        print("  → This is what post-hoc analysis cannot access.")
    else:
        print("  → Temporal information is not strongly encoded.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    results = {
        'epochs': list(epochs),
        'r2_history': r2_history,
        'silhouette_history': sil_history,
        'final_r2': r2_history[-1],
        'epoch_prediction_r2': epoch_r2,
    }

    with open('journal/trajectory_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nR² evolution: {r2_history[0]:.4f} → {r2_history[-1]:.4f}")
    print(f"The observer learned to access carry_count with increasing fidelity")
    print(f"as it observed more of the training process.")

    if epoch_r2 > 0.5:
        print(f"\nCRITICAL: The latent space encodes epoch information (R²={epoch_r2:.4f})")
        print(f"This temporal signature is what distinguishes online from post-hoc.")

    return results


if __name__ == "__main__":
    analyze_trajectory()
