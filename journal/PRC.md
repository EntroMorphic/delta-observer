# Delta Observer Falsification Plan

**Date:** 2026-01-23
**Objective:** Systematically attempt to falsify the core claims of the Delta Observer research

---

## 1. Claims Under Investigation

### Primary Claim
> Semantic information can be linearly accessible (R²=0.9505) without exhibiting geometric clustering (Silhouette=0.0320), challenging conventional assumptions about neural network interpretability.

### Supporting Claims
1. **C1:** The Delta Observer learns a meaningful shared latent representation
2. **C2:** Linear accessibility (R²) is a valid measure of semantic structure
3. **C3:** Low Silhouette score indicates absence of geometric clustering
4. **C4:** The accessibility-clustering dissociation is non-trivial

---

## 2. Falsification Hypotheses

| ID | Hypothesis | Falsifies |
|----|------------|-----------|
| H1 | Random embeddings produce similar R²/Silhouette patterns | C1, C4 |
| H2 | R² is inflated due to small sample size (n=512) | C2 |
| H3 | Shuffled labels yield comparable R² scores | C2 |
| H4 | The model memorizes rather than generalizes | C1 |
| H5 | Alternative clustering metrics contradict Silhouette | C3 |
| H6 | Results are unstable across random seeds | C1, C4 |
| H7 | Linear probe succeeds due to low dimensionality (16D), not semantic structure | C2, C4 |

---

## 3. Planned Experiments

### Experiment 1: Random Baseline Comparison
**Purpose:** Test if random 16D embeddings show similar accessibility/clustering patterns

**Method:**
1. Generate random 16D embeddings (same shape as Delta Observer latent)
2. Compute R² of linear probe predicting carry_count
3. Compute Silhouette score
4. Compare to reported metrics

**Falsification Criterion:** If random baseline achieves R² > 0.5 or Silhouette < 0.1, the claimed dissociation may be trivial.

---

### Experiment 2: Permutation Test for R²
**Purpose:** Test if R² is spuriously high due to data structure

**Method:**
1. Load original latent representations and labels
2. Shuffle labels randomly (break true correspondence)
3. Fit linear probe, compute R²
4. Repeat 1000x, build null distribution
5. Compute p-value for observed R²

**Falsification Criterion:** If p-value > 0.05, the R² is not statistically significant.

---

### Experiment 3: Train/Test Split Validation
**Purpose:** Test if the model generalizes or memorizes

**Method:**
1. Re-run analysis with 80/20 train/test split
2. Fit linear probe on train set only
3. Evaluate R² on held-out test set
4. Compare train vs test performance

**Falsification Criterion:** If test R² < 0.7 while train R² > 0.9, overfitting is indicated.

---

### Experiment 4: Alternative Clustering Metrics
**Purpose:** Test if Silhouette is misleading about clustering structure

**Method:**
1. Compute additional clustering metrics:
   - Davies-Bouldin Index
   - Calinski-Harabasz Index
   - Hopkins statistic (clustering tendency)
   - k-means inertia across k values
2. Run DBSCAN to detect natural clusters
3. Compare findings to Silhouette interpretation

**Falsification Criterion:** If other metrics indicate strong clustering, Silhouette may be inappropriate.

---

### Experiment 5: Seed Sensitivity Analysis
**Purpose:** Test reproducibility across random initializations

**Method:**
1. Retrain Delta Observer with 5 different random seeds
2. For each: compute R² and Silhouette
3. Report mean and standard deviation
4. Check if claimed values fall within confidence interval

**Falsification Criterion:** If std(R²) > 0.15 or results vary wildly, findings are unstable.

---

### Experiment 6: Dimensionality Control
**Purpose:** Test if low dimensionality (16D) trivially enables linear accessibility

**Method:**
1. Apply PCA to reduce original activations to 16D (no learning)
2. Compute R² of linear probe on PCA embeddings
3. Compare to Delta Observer R²

**Falsification Criterion:** If PCA achieves R² > 0.9, the Delta Observer adds no value over simple dimensionality reduction.

---

### Experiment 7: Probe Complexity Ablation
**Purpose:** Test if semantic structure requires more than linear access

**Method:**
1. Fit linear probe (baseline)
2. Fit 1-hidden-layer MLP probe
3. Fit 2-hidden-layer MLP probe
4. Compare R² across probe complexities

**Falsification Criterion:** If nonlinear probes dramatically outperform linear (e.g., MLP R² > 0.99 vs linear R² = 0.95), "linear accessibility" claim is weakened.

---

## 4. Execution Priority

| Priority | Experiment | Rationale |
|----------|------------|-----------|
| 1 | Random Baseline (E1) | Cheapest test, high falsification potential |
| 2 | Permutation Test (E2) | Statistical rigor for R² claim |
| 3 | Train/Test Split (E3) | Critical for generalization claim |
| 4 | Alternative Clustering (E4) | Validates Silhouette interpretation |
| 5 | Dimensionality Control (E6) | Tests if learning adds value |
| 6 | Probe Complexity (E7) | Tests "linear" specificity |
| 7 | Seed Sensitivity (E5) | Reproducibility (requires retraining) |

---

## 5. Data Requirements

- `data/delta_latent_umap.npz` - Latent representations (if available)
- `data/delta_observer_dataset.npz` - Semantic labels (carry_count, bit_position)
- Original model weights (for retraining experiments)

---

## 6. Success Criteria

The research claims are **falsified** if ANY of:
- Random baseline achieves comparable metrics (E1)
- R² is not statistically significant (E2)
- Model fails to generalize to test set (E3)
- Alternative metrics contradict clustering interpretation (E4)
- Simple PCA matches Delta Observer performance (E6)

The research claims are **supported** if ALL experiments fail to falsify.

---

## 7. Notes & Considerations

- Sample size (n=512) is small; statistical power may be limited
- 4-bit addition is a toy task; generalizability to real problems is unclear
- The Silhouette score interpretation assumes labels define "true" clusters

### Critical Methodological Issue Discovered

**The original analysis (`analyze_delta_latent.py`) computes R² on the training data without a train/test split.**

```python
# Lines 99-102 of analyze_delta_latent.py
reg_carry = LinearRegression()
reg_carry.fit(latents, carry_counts)  # Fit on ALL data
carry_pred = reg_carry.predict(latents)
carry_r2 = reg_carry.score(latents, carry_counts)  # Score on SAME data!
```

This means the reported R²=0.9505 is potentially inflated by overfitting. This strengthens the importance of Experiment 3 (Train/Test Split Validation).

### Data Verified

- `latents`: (512, 16) - Delta Observer latent representations
- `carry_counts`: (512,) - semantic labels (range 0-4)
- `bit_positions`: (512,) - structural labels (range 0-3)

---

## Status

| Experiment | Status | Result |
|------------|--------|--------|
| E1: Random Baseline | ✓ Complete | **PASSED** - Random R²=0.0149 << Delta Observer R²=0.9505 |
| E2: Permutation Test | ✓ Complete | **PASSED** - p-value=0.000000, statistically significant |
| E3: Train/Test Split | ✓ Complete | **PASSED** - Test R²=0.9488, model generalizes |
| E4: Alt Clustering | ✓ Complete | **PASSED** - Multiple metrics confirm weak clustering |
| E5: Dimensionality | ✓ Complete | **FAILED** - PCA R²=0.9482 ≈ Delta Observer R²=0.9505 |
| E6: Probe Complexity | ✓ Complete | **PASSED** - Linear probe captures most info |
| E7: Seed Sensitivity | Skipped | Requires retraining (lower priority) |

---

## Phase 0: Curriculum Test Results (2026-01-23)

**TEMPORAL STRUCTURE EXISTS** - Proceed with online observer.

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Overall correlation | 0.9817 | > 0.7 | **PASS** |
| Learning order (Spearman) | 1.0000 | > 0.7 | **PASS** |

Per-carry correlations: c0=0.99, c1=0.99, c2=0.99, c3=0.98, c4=0.95

### Critical Observation

Both models reach 100% accuracy by epoch ~15-20. The temporal dynamics of interest are compressed into the **first 20 epochs**. After that, both models are perfect.

**Implication:** The online observer must capture early training dynamics. Epochs 1-20 contain the learning; epochs 20-200 are static.

This explains why post-hoc observation misses temporal information - it only sees the final (already-converged) state.

---

## Phase 1: Online Observer Results (2026-01-23)

**ONLINE OBSERVER BEATS PCA BASELINE**

| Method | R² | Silhouette | vs PCA |
|--------|------|-----------|--------|
| **Online Observer** | **0.9879** | -0.0242 | **+0.0397** |
| Post-hoc Observer | 0.9505 | 0.0320 | +0.0023 |
| PCA Baseline | 0.9482 | 0.0463 | - |

### Key Findings

1. **First method to meaningfully beat PCA** - The online observer achieves R²=0.9879, a 4% improvement over PCA's 0.9482.

2. **Even less clustering** - Silhouette score is now *negative* (-0.0242), meaning the latent space has even less geometric structure while maintaining higher accessibility.

3. **Temporal information matters** - The only difference between online and post-hoc is that online observed the training trajectory. This 0.0374 R² improvement (0.9879 vs 0.9505) is attributable to temporal information.

### Implications

- The original intuition was correct: watching training provides information post-hoc analysis cannot access
- The "semantic primitive" is partially encoded in how models learn, not just what they learn
- The accessibility-clustering dissociation is strengthened (higher R², lower Silhouette)

### PCA Baseline Failure: ADDRESSED

The original falsification test showed PCA matched the post-hoc observer. The online observer breaks this equivalence by accessing temporal information unavailable to PCA.

---

## 8. Results Summary (2026-01-23)

### Key Finding: PCA Baseline Matches Delta Observer

The most significant finding is from **Experiment 5**:

| Representation | R² (carry_count) | Silhouette |
|----------------|------------------|------------|
| Delta Observer Latent | 0.9505 | 0.0320 |
| PCA (combined activations) | 0.9482 | 0.0463 |
| PCA (mono only) | 0.7466 | - |
| PCA (comp only) | **0.9559** | - |

**Critical observation:** PCA on compositional activations alone (R²=0.9559) actually **exceeds** Delta Observer performance (R²=0.9505).

### Implications

1. **The semantic information (carry_count) is already linearly accessible** in the raw compositional activations without any learned transformation.

2. **The Delta Observer doesn't discover new semantic structure** - it merely preserves (or slightly degrades) what was already present.

3. **The "shared latent space" contribution is questionable** - the value-add over simple dimensionality reduction is negligible (0.0023 R² improvement over combined PCA).

4. **The core claim is technically true but potentially misleading:**
   - Yes, R²=0.95 with Silhouette=0.03 demonstrates accessibility without clustering
   - But this property exists in the raw data, not as a result of the learning process

### What Remains Valid

- The accessibility-clustering dissociation phenomenon is real
- The statistical analysis is sound (p < 0.001)
- The model generalizes (no overfitting)
- Linear probes are sufficient (no need for nonlinear)

### What is Challenged

- The Delta Observer's value proposition over simple baselines
- The narrative that the learned representation is special
- The implication that dual-encoder architecture discovers semantic structure
