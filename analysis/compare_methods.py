#!/usr/bin/env python3
"""
Compare Methods: Online vs Post-hoc vs PCA
===========================================
Runs all three methods and produces a comparison table.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import json


def compare_all_methods():
    print("=" * 70)
    print("METHOD COMPARISON: Online vs Post-hoc vs PCA")
    print("=" * 70)

    results = {}

    # Load online observer data
    try:
        online_data = np.load('data/online_observer_latents.npz')
        online_latents = online_data['latents']
        carry_counts = online_data['carry_counts']

        reg = LinearRegression()
        reg.fit(online_latents, carry_counts)
        online_r2 = reg.score(online_latents, carry_counts)
        online_sil = silhouette_score(online_latents, carry_counts)

        results['online'] = {'r2': online_r2, 'silhouette': online_sil}
        print(f"\n[Online Observer]")
        print(f"  R²:         {online_r2:.4f}")
        print(f"  Silhouette: {online_sil:.4f}")
    except FileNotFoundError:
        print("\n[Online Observer] Data not found. Run: python models/delta_observer.py")
        results['online'] = None

    # Load post-hoc observer data
    try:
        posthoc_data = np.load('data/delta_latent_umap.npz')
        posthoc_latents = posthoc_data['latents']
        posthoc_carry = posthoc_data['carry_counts']

        reg = LinearRegression()
        reg.fit(posthoc_latents, posthoc_carry)
        posthoc_r2 = reg.score(posthoc_latents, posthoc_carry)
        posthoc_sil = silhouette_score(posthoc_latents, posthoc_carry)

        results['posthoc'] = {'r2': posthoc_r2, 'silhouette': posthoc_sil}
        print(f"\n[Post-hoc Observer]")
        print(f"  R²:         {posthoc_r2:.4f}")
        print(f"  Silhouette: {posthoc_sil:.4f}")
    except FileNotFoundError:
        print("\n[Post-hoc Observer] Data not found. Run legacy pipeline.")
        results['posthoc'] = None

    # PCA baseline
    try:
        orig_data = np.load('data/delta_observer_dataset.npz')
        mono_act = orig_data['mono_activations']
        comp_act = orig_data['comp_activations']
        pca_carry = orig_data['carry_counts']

        combined = np.concatenate([mono_act, comp_act], axis=1)
        pca = PCA(n_components=16)
        pca_latents = pca.fit_transform(combined)

        reg = LinearRegression()
        reg.fit(pca_latents, pca_carry)
        pca_r2 = reg.score(pca_latents, pca_carry)
        pca_sil = silhouette_score(pca_latents, pca_carry)

        results['pca'] = {'r2': pca_r2, 'silhouette': pca_sil}
        print(f"\n[PCA Baseline]")
        print(f"  R²:         {pca_r2:.4f}")
        print(f"  Silhouette: {pca_sil:.4f}")
    except FileNotFoundError:
        print("\n[PCA Baseline] Data not found.")
        results['pca'] = None

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<25} {'R²':>12} {'Silhouette':>12} {'Δ vs PCA':>12}")
    print("-" * 61)

    pca_r2_val = results['pca']['r2'] if results['pca'] else 0

    for name, key in [('Online Observer', 'online'),
                      ('Post-hoc Observer', 'posthoc'),
                      ('PCA Baseline', 'pca')]:
        if results[key]:
            r2 = results[key]['r2']
            sil = results[key]['silhouette']
            delta = r2 - pca_r2_val if key != 'pca' else 0
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}" if delta < 0 else "—"
            print(f"{name:<25} {r2:>12.4f} {sil:>12.4f} {delta_str:>12}")
        else:
            print(f"{name:<25} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    # Verdict
    print("\n" + "-" * 61)
    if results['online'] and results['pca']:
        improvement = results['online']['r2'] - results['pca']['r2']
        if improvement > 0.01:
            print(f"✓ Online Observer beats PCA by {improvement:.2%}")
        else:
            print(f"~ Online Observer ≈ PCA (Δ = {improvement:.4f})")

    # Save results
    save_results = {k: v for k, v in results.items() if v is not None}
    with open('journal/method_comparison.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to journal/method_comparison.json")

    return results


if __name__ == "__main__":
    compare_all_methods()
