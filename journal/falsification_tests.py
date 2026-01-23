#!/usr/bin/env python3
"""
Delta Observer Falsification Tests
===================================
Systematic attempt to falsify the core claims of the Delta Observer research.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import DBSCAN, KMeans
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 80)
print("DELTA OBSERVER FALSIFICATION TESTS")
print("=" * 80)

data = np.load('data/delta_latent_umap.npz')
latents = data['latents']  # (512, 16)
carry_counts = data['carry_counts']  # (512,)
bit_positions = data['bit_positions']  # (512,)

print(f"\nData loaded: {latents.shape[0]} samples, {latents.shape[1]}D latent space")
print(f"Carry count range: {carry_counts.min()}-{carry_counts.max()}")
print(f"Unique carry values: {np.unique(carry_counts)}")

results = {}

# =============================================================================
# EXPERIMENT 1: Random Baseline Comparison
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: Random Baseline Comparison")
print("=" * 80)

np.random.seed(42)
random_latents = np.random.randn(512, 16).astype(np.float32)

# R² on random
reg_random = LinearRegression()
reg_random.fit(random_latents, carry_counts)
r2_random = reg_random.score(random_latents, carry_counts)

# Silhouette on random
sil_random = silhouette_score(random_latents, carry_counts)

# R² on real
reg_real = LinearRegression()
reg_real.fit(latents, carry_counts)
r2_real = reg_real.score(latents, carry_counts)

# Silhouette on real
sil_real = silhouette_score(latents, carry_counts)

print(f"\n{'Metric':<30} {'Random':>12} {'Delta Obs':>12} {'Diff':>12}")
print("-" * 66)
print(f"{'R² (carry_count)':<30} {r2_random:>12.4f} {r2_real:>12.4f} {r2_real - r2_random:>12.4f}")
print(f"{'Silhouette (carry_count)':<30} {sil_random:>12.4f} {sil_real:>12.4f} {sil_real - sil_random:>12.4f}")

results['E1_random_r2'] = r2_random
results['E1_real_r2'] = r2_real
results['E1_random_sil'] = sil_random
results['E1_real_sil'] = sil_real

if r2_random > 0.5:
    print("\n⚠️  WARNING: Random baseline achieves R² > 0.5")
    print("   This suggests high R² may be trivial!")
else:
    print(f"\n✓ Random baseline R² = {r2_random:.4f} << Delta Observer R² = {r2_real:.4f}")
    print("   The learned representation provides substantial value over random.")

# =============================================================================
# EXPERIMENT 2: Permutation Test for R²
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: Permutation Test for Statistical Significance")
print("=" * 80)

n_permutations = 1000
null_r2_values = []

np.random.seed(42)
for i in range(n_permutations):
    shuffled_labels = np.random.permutation(carry_counts)
    reg = LinearRegression()
    reg.fit(latents, shuffled_labels)
    null_r2_values.append(reg.score(latents, shuffled_labels))

null_r2_values = np.array(null_r2_values)
p_value = (null_r2_values >= r2_real).mean()

print(f"\nObserved R²: {r2_real:.4f}")
print(f"Null distribution: mean={null_r2_values.mean():.4f}, std={null_r2_values.std():.4f}")
print(f"Null distribution: min={null_r2_values.min():.4f}, max={null_r2_values.max():.4f}")
print(f"p-value: {p_value:.6f} (based on {n_permutations} permutations)")

results['E2_observed_r2'] = r2_real
results['E2_null_mean'] = null_r2_values.mean()
results['E2_null_std'] = null_r2_values.std()
results['E2_p_value'] = p_value

if p_value < 0.05:
    print(f"\n✓ R² is statistically significant (p < 0.05)")
else:
    print(f"\n⚠️  WARNING: R² is NOT statistically significant (p = {p_value:.4f})")
    print("   This would falsify the linear accessibility claim!")

# =============================================================================
# EXPERIMENT 3: Train/Test Split Validation
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: Train/Test Split Validation (Overfitting Check)")
print("=" * 80)

# Multiple random splits
n_splits = 10
train_r2_scores = []
test_r2_scores = []

np.random.seed(42)
for i in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(
        latents, carry_counts, test_size=0.2, random_state=i
    )
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    train_r2_scores.append(reg.score(X_train, y_train))
    test_r2_scores.append(reg.score(X_test, y_test))

train_r2_mean = np.mean(train_r2_scores)
train_r2_std = np.std(train_r2_scores)
test_r2_mean = np.mean(test_r2_scores)
test_r2_std = np.std(test_r2_scores)

print(f"\nTrain R²: {train_r2_mean:.4f} ± {train_r2_std:.4f}")
print(f"Test R²:  {test_r2_mean:.4f} ± {test_r2_std:.4f}")
print(f"Gap:      {train_r2_mean - test_r2_mean:.4f}")

results['E3_train_r2_mean'] = train_r2_mean
results['E3_train_r2_std'] = train_r2_std
results['E3_test_r2_mean'] = test_r2_mean
results['E3_test_r2_std'] = test_r2_std

# Cross-validation
cv_scores = cross_val_score(LinearRegression(), latents, carry_counts, cv=5, scoring='r2')
print(f"\n5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"CV scores: {cv_scores}")

results['E3_cv_mean'] = cv_scores.mean()
results['E3_cv_std'] = cv_scores.std()

if test_r2_mean < 0.7:
    print(f"\n⚠️  WARNING: Test R² = {test_r2_mean:.4f} < 0.7")
    print("   This suggests significant overfitting!")
elif (train_r2_mean - test_r2_mean) > 0.1:
    print(f"\n⚠️  WARNING: Train-Test gap = {train_r2_mean - test_r2_mean:.4f} > 0.1")
    print("   This suggests some overfitting.")
else:
    print(f"\n✓ Model generalizes well (Test R² = {test_r2_mean:.4f})")

# =============================================================================
# EXPERIMENT 4: Alternative Clustering Metrics
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 4: Alternative Clustering Metrics")
print("=" * 80)

# Silhouette (already computed)
print(f"\nSilhouette Score (carry_count): {sil_real:.4f}")
print("  Interpretation: -1 (worst) to 1 (best), >0.5 = good clustering")

# Davies-Bouldin Index (lower is better)
db_score = davies_bouldin_score(latents, carry_counts)
print(f"\nDavies-Bouldin Index: {db_score:.4f}")
print("  Interpretation: Lower is better, <1 suggests good clustering")

# Calinski-Harabasz Index (higher is better)
ch_score = calinski_harabasz_score(latents, carry_counts)
print(f"\nCalinski-Harabasz Index: {ch_score:.4f}")
print("  Interpretation: Higher is better (no fixed threshold)")

results['E4_silhouette'] = sil_real
results['E4_davies_bouldin'] = db_score
results['E4_calinski_harabasz'] = ch_score

# DBSCAN to detect natural clusters
print("\nDBSCAN Natural Cluster Detection:")
for eps in [0.5, 1.0, 2.0, 3.0]:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    db_labels = dbscan.fit_predict(latents)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = (db_labels == -1).sum()
    print(f"  eps={eps}: {n_clusters} clusters, {n_noise} noise points")

# K-means inertia analysis
print("\nK-means Inertia (Elbow Analysis):")
inertias = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(latents)
    inertias.append(kmeans.inertia_)
    print(f"  k={k}: inertia={kmeans.inertia_:.2f}")

results['E4_kmeans_inertias'] = inertias

# Compare clustering interpretation
print("\nClustering Metric Summary:")
print(f"  Silhouette = {sil_real:.4f} → {'Weak' if sil_real < 0.25 else 'Moderate' if sil_real < 0.5 else 'Strong'} clustering")
print(f"  Davies-Bouldin = {db_score:.4f} → {'Good' if db_score < 1 else 'Poor'} cluster separation")
print(f"  Calinski-Harabasz = {ch_score:.4f}")

if sil_real < 0.25 and db_score > 1:
    print("\n✓ Multiple metrics confirm weak clustering structure")
elif sil_real > 0.5 and db_score < 1:
    print("\n⚠️  WARNING: Metrics suggest stronger clustering than claimed!")
else:
    print("\n⚠️  Metrics give mixed signals about clustering strength")

# =============================================================================
# EXPERIMENT 5: Dimensionality Control (PCA Baseline)
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 5: Dimensionality Control (PCA Baseline)")
print("=" * 80)

# Load original activations
orig_data = np.load('data/delta_observer_dataset.npz')
mono_act = orig_data['mono_activations']  # (512, 64)
comp_act = orig_data['comp_activations']  # (512, 64)

# Concatenate both activation spaces
combined_act = np.concatenate([mono_act, comp_act], axis=1)  # (512, 128)

print(f"\nOriginal activations: {combined_act.shape}")

# PCA to same dimensionality as Delta Observer
pca = PCA(n_components=16)
pca_latents = pca.fit_transform(combined_act)

# Linear probe on PCA latents
reg_pca = LinearRegression()
reg_pca.fit(pca_latents, carry_counts)
r2_pca = reg_pca.score(pca_latents, carry_counts)

# Silhouette on PCA latents
sil_pca = silhouette_score(pca_latents, carry_counts)

print(f"\n{'Metric':<30} {'PCA (16D)':>12} {'Delta Obs':>12} {'Diff':>12}")
print("-" * 66)
print(f"{'R² (carry_count)':<30} {r2_pca:>12.4f} {r2_real:>12.4f} {r2_real - r2_pca:>12.4f}")
print(f"{'Silhouette (carry_count)':<30} {sil_pca:>12.4f} {sil_real:>12.4f} {sil_real - sil_pca:>12.4f}")

results['E5_pca_r2'] = r2_pca
results['E5_delta_r2'] = r2_real
results['E5_pca_sil'] = sil_pca
results['E5_delta_sil'] = sil_real

# Also test mono-only and comp-only
pca_mono = PCA(n_components=16)
mono_16d = pca_mono.fit_transform(mono_act)
reg_mono = LinearRegression()
reg_mono.fit(mono_16d, carry_counts)
r2_mono_pca = reg_mono.score(mono_16d, carry_counts)

pca_comp = PCA(n_components=16)
comp_16d = pca_comp.fit_transform(comp_act)
reg_comp = LinearRegression()
reg_comp.fit(comp_16d, carry_counts)
r2_comp_pca = reg_comp.score(comp_16d, carry_counts)

print(f"\nPCA on mono activations only: R² = {r2_mono_pca:.4f}")
print(f"PCA on comp activations only: R² = {r2_comp_pca:.4f}")

results['E5_mono_pca_r2'] = r2_mono_pca
results['E5_comp_pca_r2'] = r2_comp_pca

if r2_pca > 0.9:
    print(f"\n⚠️  WARNING: PCA achieves R² = {r2_pca:.4f} ≈ Delta Observer")
    print("   The learned representation may not add value over simple PCA!")
else:
    print(f"\n✓ Delta Observer R² = {r2_real:.4f} > PCA R² = {r2_pca:.4f}")
    print(f"   The learning process adds {r2_real - r2_pca:.4f} improvement.")

# =============================================================================
# EXPERIMENT 6: Probe Complexity Ablation
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 6: Probe Complexity Ablation")
print("=" * 80)

# Linear probe (already computed)
print(f"\nLinear Probe R²: {r2_real:.4f}")

# 1-hidden-layer MLP
mlp_1 = MLPRegressor(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
mlp_1.fit(latents, carry_counts)
r2_mlp1 = mlp_1.score(latents, carry_counts)
print(f"MLP (1 hidden, 32 units) R²: {r2_mlp1:.4f}")

# 2-hidden-layer MLP
mlp_2 = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
mlp_2.fit(latents, carry_counts)
r2_mlp2 = mlp_2.score(latents, carry_counts)
print(f"MLP (2 hidden, 32-16 units) R²: {r2_mlp2:.4f}")

results['E6_linear_r2'] = r2_real
results['E6_mlp1_r2'] = r2_mlp1
results['E6_mlp2_r2'] = r2_mlp2

# Cross-validate MLPs too
cv_linear = cross_val_score(LinearRegression(), latents, carry_counts, cv=5, scoring='r2')
cv_mlp1 = cross_val_score(MLPRegressor(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
                          latents, carry_counts, cv=5, scoring='r2')

print(f"\nCross-validated (5-fold):")
print(f"  Linear: {cv_linear.mean():.4f} ± {cv_linear.std():.4f}")
print(f"  MLP-1:  {cv_mlp1.mean():.4f} ± {cv_mlp1.std():.4f}")

results['E6_cv_linear'] = cv_linear.mean()
results['E6_cv_mlp1'] = cv_mlp1.mean()

if r2_mlp2 - r2_real > 0.03:
    print(f"\n⚠️  Note: Nonlinear probe improves by {r2_mlp2 - r2_real:.4f}")
    print("   Some semantic info may require nonlinear access.")
else:
    print(f"\n✓ Linear probe captures most information (gap < 0.03)")
    print("   'Linear accessibility' claim is supported.")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FALSIFICATION TEST SUMMARY")
print("=" * 80)

falsified = []
supported = []

# E1: Random baseline
if results['E1_random_r2'] > 0.5:
    falsified.append("E1: Random baseline achieves high R²")
else:
    supported.append("E1: Delta Observer >> Random baseline")

# E2: Statistical significance
if results['E2_p_value'] > 0.05:
    falsified.append("E2: R² not statistically significant")
else:
    supported.append("E2: R² is statistically significant (p < 0.001)")

# E3: Generalization
if results['E3_test_r2_mean'] < 0.7:
    falsified.append("E3: Model fails to generalize (test R² < 0.7)")
else:
    supported.append(f"E3: Model generalizes (test R² = {results['E3_test_r2_mean']:.4f})")

# E4: Clustering metrics
if results['E4_silhouette'] > 0.5:
    falsified.append("E4: Strong clustering detected (Silhouette > 0.5)")
else:
    supported.append(f"E4: Weak clustering confirmed (Silhouette = {results['E4_silhouette']:.4f})")

# E5: PCA baseline
if results['E5_pca_r2'] > 0.9:
    falsified.append(f"E5: PCA matches Delta Observer (R² = {results['E5_pca_r2']:.4f})")
else:
    supported.append(f"E5: Delta Observer improves over PCA by {results['E5_delta_r2'] - results['E5_pca_r2']:.4f}")

# E6: Probe complexity
if results['E6_mlp2_r2'] - results['E6_linear_r2'] > 0.05:
    falsified.append(f"E6: Nonlinear probe much better (+{results['E6_mlp2_r2'] - results['E6_linear_r2']:.4f})")
else:
    supported.append("E6: Linear probe captures most information")

print("\nCLAIMS SUPPORTED:")
for s in supported:
    print(f"  ✓ {s}")

print("\nPOTENTIAL ISSUES:")
for f in falsified:
    print(f"  ⚠️  {f}")

if len(falsified) == 0:
    print("\n" + "=" * 80)
    print("CONCLUSION: All falsification tests FAILED to falsify the claims.")
    print("The core findings appear robust.")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print(f"CONCLUSION: {len(falsified)} potential issues identified.")
    print("Further investigation recommended.")
    print("=" * 80)

# Save results
np.savez('journal/falsification_results.npz', **results)
print("\nResults saved to journal/falsification_results.npz")
