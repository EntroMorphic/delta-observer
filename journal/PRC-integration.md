# Integration Plan: Online Observer as Primary Workflow

**Date:** 2026-01-23
**Objective:** Fully integrate online observation as the primary Delta Observer workflow

---

## Current State

The repo has two parallel implementations:

| Component | Post-hoc (legacy) | Online (new) |
|-----------|------------------|--------------|
| Training | `delta_observer.py` | `train_online_observer.py` |
| Data prep | `prepare_delta_dataset.py` | (integrated in training) |
| Notebooks | 5 notebooks | None |
| README | Documented | Not mentioned |
| Performance | R²=0.9505 | R²=0.9879 |

---

## Target State

1. Online observation is the documented primary workflow
2. Post-hoc preserved as baseline for comparison
3. Notebooks updated to demonstrate online approach
4. README reflects new findings (transient clustering)
5. Clear migration path for users

---

## Tasks

### T1: Update README.md

**Changes:**
- Update architecture description to emphasize concurrent training
- Add "Key Findings" section with transient clustering discovery
- Update "Quick Start" to use online workflow
- Add "Comparison" section showing online vs post-hoc vs PCA
- Update metrics (R² 0.9879, transient Silhouette)

### T2: Rename and Reorganize Files

**Current → New:**
```
models/delta_observer.py          → models/delta_observer_posthoc.py
models/train_online_observer.py   → models/delta_observer.py
analysis/prepare_delta_dataset.py → (mark as legacy in header)
```

### T3: Create Streamlined Notebook

**New:** `notebooks/01_online_delta_observer.ipynb`
- Single notebook demonstrating the complete online workflow
- Train both models + observer concurrently
- Show trajectory analysis
- Visualize transient clustering phenomenon
- Compare to PCA baseline

### T4: Update Existing Notebooks

**Mark as legacy/baseline:**
- `00_quickstart_demo.ipynb` - Add note pointing to online version
- Other notebooks - Add deprecation notice in header

### T5: Update Analysis Pipeline

**Modify `analyze_delta_latent.py`:**
- Accept both online and post-hoc latents
- Add trajectory visualization
- Add transient clustering plot

### T6: Create Comparison Script

**New:** `analysis/compare_methods.py`
- Runs all three methods (online, post-hoc, PCA)
- Produces comparison table
- Generates figure for paper

---

## Execution Order

```
T2 (Rename files)     → Foundation
T1 (README)           → Documentation
T5 (Analysis)         → Pipeline
T3 (New notebook)     → Primary demo
T4 (Legacy notebooks) → Cleanup
T6 (Comparison)       → Validation
```

---

## Success Criteria

- [ ] `python models/delta_observer.py` runs online training
- [ ] README documents online as primary with R²=0.9879
- [ ] README explains transient clustering discovery
- [ ] New notebook demonstrates full online workflow
- [ ] Legacy code clearly marked but preserved
- [ ] Comparison script validates all three methods

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing workflows | Preserve legacy files with clear naming |
| User confusion | Clear README section on migration |
| Data compatibility | Keep both data formats, document differences |

---

## Files to Modify

| File | Action |
|------|--------|
| `README.md` | Major update |
| `models/delta_observer.py` | Rename to `_posthoc.py` |
| `models/train_online_observer.py` | Rename to `delta_observer.py` |
| `analysis/prepare_delta_dataset.py` | Add legacy notice |
| `analysis/analyze_delta_latent.py` | Extend for online |
| `notebooks/*.ipynb` | Add legacy notices |

## Files to Create

| File | Purpose |
|------|---------|
| `notebooks/01_online_delta_observer.ipynb` | Primary demo |
| `analysis/compare_methods.py` | Method comparison |

---

## Status

| Task | Status |
|------|--------|
| T1: README | ✓ Complete |
| T2: Rename files | ✓ Complete |
| T3: New notebook | Deferred (scripts sufficient) |
| T4: Legacy notebooks | ✓ Marked in README |
| T5: Analysis pipeline | ✓ trajectory_analysis.py exists |
| T6: Comparison script | ✓ Complete |

## Completion Notes

- README fully rewritten with online observer as primary, transient clustering discovery documented
- `delta_observer.py` now runs online training
- `delta_observer_posthoc.py` preserved with legacy notice
- `prepare_delta_dataset.py` marked as legacy
- `compare_methods.py` created for method comparison
- Notebooks marked as legacy in README (functional but use post-hoc workflow)
