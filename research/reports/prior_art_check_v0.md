# Prior-art check v0 (2026-10-27)

**Scope.** A focused web search of about one hour on the candidate findings in
[`interface_sweep_v0.md`](interface_sweep_v0.md) and
[`contract_experiments_v0.md`](contract_experiments_v0.md). It is not a
systematic review. "Not found" means not found in these searches, which is
not proof of novelty.

## Verdicts

| Candidate | Closest prior art | Verdict |
|---|---|---|
| Locality and width separated as causes of staleness tolerance | DGL; boundary compression; PETRA | **Plausibly open, narrow** |
| Staleness harm is directional (permuted-error control) | SDDE staleness analysis; anisotropic noise; optimizer-dependent delay robustness | **Plausibly open, small**, and likely optimizer-dependent |
| Interface lifecycle: harm peaks at the transition, then settles | Post-local SGD; critical periods; convergence-triggered layer freezing | **Mostly known.** The stalled "half-formed interface" observation was not found |
| Correlated perturbations aid fitting but hurt generalization | Anticorrelated noise injection | **Known in direction** (consistent with it) |
| Per-interface triggers lose to a global switch | Freezing literature warns against premature freezing | **Consistent with known cautions** |

## Details

### 1. Locality versus width

**Decoupled greedy learning** (Belilovsky et al., ICML 2020; extended
2021):
- Modules train on local auxiliary losses and receive no backward signal.
- The asynchronous version tolerates simulated slowdowns. Accuracy holds up
  to about 1.2× and the losses stay small up to 2×.
- It does not compare with end-to-end pipelines under the same delays.

**Compression at pipeline boundaries** studies bandwidth, not staleness:
- ResBM (2026) uses bottlenecks for 128× compression.
- SWARM (ICML 2023) trains asynchronously with 8-bit activations.
- An MLSys 2024 study finds that compressing early layers hurts most.

**PETRA** (ICLR 2025) reports that local-learning methods "often lead to
unrecoverable performance drops", and proposes reversible stages instead.

**Not found:** a controlled experiment that separates locality (local inputs
and outputs) from message width as causes of staleness tolerance, under
end-to-end backprop. Our setup also differs from DGL: gradients still flow
across modules, and the local outputs are the true task targets rather than
auxiliary heads.

**Caveat.** Our "locality" includes per-stage *inputs*, which ordinary deep
networks lack. That limits how far the result transfers.

### 2. Directional staleness harm

**Related work, none of it testing this directly:**
- Yu, Chen and Poor (IEEE TSP; arXiv 2406.11159) model staleness as a
  stochastic delay differential equation whose behaviour depends on the
  Hessian eigenvalues. This is theory, with no comparison against noise.
- Zhu et al. (ICML 2019) compare anisotropic and isotropic noise at matched
  norm. They find noise aligned with the curvature helps *escape sharp
  minima*. Our result goes the other way for fitting: aligned staleness error
  was worse than misaligned.
- Zmushko et al. (arXiv 2606.30634, June 2026) report that a one-step gradient
  delay severely degrades AdamW but not Muon. They attribute the harm to the
  optimizer, not to delay itself.

**Not found:** a control that keeps the staleness error's norm, entries and
correlation over time while destroying its alignment with the network. That
control appears to be new and is reusable.

**Implication.** Our result used Adam, and the Zmushko paper suggests the
directional harm may depend on the optimizer. A confirmatory test should
include SGD and Muon.

### 3. Interface lifecycle

**Known at the global level:**
- "Sync early, async later": post-local SGD (Lin et al., 2020), and
  DiLoCo-style warm starts.
- Critical learning periods (Achille et al., 2019).

**Known at the per-layer level:**
- Convergence-triggered freezing: Egeria (training plasticity against a
  reference model), AutoFreeze (rate of change of gradient norms), SmartFRZ
  (CKA against a trained reference).
- Egeria is explicitly designed to "avoid premature freezing", which is the
  same failure our per-interface triggers showed.
- A 2026 study of progressive freezing finds layerwise schedules more robust
  than global ones. That is the opposite of our X2 result, in a different
  setting (binarised networks).

**Not found:** stalled runs traced to a single interface whose meaning formed
only partly, while the other interfaces are faithful.

### 4. Correlated perturbations

Orvieto et al. (ICML 2022): perturbations anticorrelated across consecutive
steps generalize better than uncorrelated ones by moving to wider minima. Our
positively correlated noise lowered generalization in the monolithic network.
That fits their picture, so our result corroborates known work rather than
adding to it.

## What remains, and is worth a confirmatory test

1. **Locality versus width,** with gradients flowing end to end. Seeds 0–9,
   both tasks, delays 0, 4 and 16. Preferably add one standard-architecture
   check, for example a residual MLP with per-block auxiliary outputs.
2. **Directional harm via the permuted-error control,** across optimizers:
   Adam, SGD with momentum, and Muon. It is new only if it survives the
   optimizer change, or if it explains the AdamW-versus-Muon difference.

Neither is large. Together they could support a short empirical note.

## Sources

- Belilovsky et al., Decoupled Greedy Learning of CNNs, ICML 2020: https://proceedings.mlr.press/v119/belilovsky20a.html ; arXiv 1901.08164 ; extended arXiv 2106.06401
- PETRA, ICLR 2025: https://arxiv.org/pdf/2406.02052
- ResBM: https://arxiv.org/pdf/2604.11947
- SWARM Parallelism, ICML 2023: https://proceedings.mlr.press/v202/ryabinin23a/ryabinin23a.pdf
- Does compressing activations help model parallel training?, MLSys 2024: https://proceedings.mlsys.org/paper_files/paper/2024/file/71381211d0abef73ed1887b83c4547b1-Supplemental-Conference.pdf
- Yu, Chen and Poor, Distributed SGD with staleness (SDDE): https://arxiv.org/abs/2406.11159
- Zhu et al., Anisotropic noise in SGD, ICML 2019: http://proceedings.mlr.press/v97/zhu19e/zhu19e.pdf
- Zmushko et al., One-step gradient delay is not a barrier: https://arxiv.org/abs/2606.30634
- PipeDream: https://arxiv.org/pdf/1806.03377
- Laskin et al., Parallel training with local updates: https://arxiv.org/abs/2012.03837
- Egeria: https://www.arxiv.org/pdf/2201.06227 ; AutoFreeze: https://www.arxiv.org/pdf/2102.01386 ; SmartFRZ: https://arxiv.org/pdf/2401.16720
- Layerwise progressive freezing: https://arxiv.org/pdf/2606.27759
- Orvieto et al., Anticorrelated noise injection, ICML 2022: https://proceedings.mlr.press/v162/orvieto22a.html
