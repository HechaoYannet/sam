# SAM Project — Progress & Handoff

**Last updated:** 2026-06-03 15:30 | **Current phase:** Phase 2 Wave 1 complete → investigating manifold structure

---

## Quick Start

```powershell
$env:PYTHONPATH = "E:\PythonProject\mm-jepa"
$env:SAM_PRETRAINED_VIT = "$env:USERPROFILE\.cache\sam\pretrained\timm\vit_tiny_patch16_224___augreg_in21k_ft_in1k"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"  # OpenMP workaround
conda run -n tct python scripts/<script>.py
```

---

## Phase 2 Wave 1 — Complete

### Color Blindness Fix ✓

**Previous hypothesis (ViT 192→256 projection) was WRONG.**

True root cause: **symbol encoder MLP bottleneck (384→128→256)** destroyed color info.
Per-category embeddings (64-dim) were well disentangled (cross-color cos ~0.1),
but the 128-dim hidden layer forced 3:1 compression; model kept shape (rewarded by
L_rel/L_analogy) and discarded color (no direct loss on full output).

Also: `ColorContrastiveLoss` had inverted semantics — same-shape-diff-color was POSITIVE.

Three fixes applied:
1. Fixed ColorContrastiveLoss: same-color-diff-shape = positive now
2. Widened MLP: hidden_dim 128→512 (77K→299K params)
3. New SymbolColorSeparationLoss: penalizes cosine >0.9 for same-shape-diff-color

| Metric | P3 Baseline | After Fix | Target |
|--------|------------|-----------|--------|
| Cross-modal color Top-1 | 16.7% | **63.3%** | >50% ✓ |
| Cross-modal color Top-3 | 53.3% | **80.0%** | — |
| Cube color retrieval | 1/6 | **6/6** | >3/6 ✓ |
| Symbol cross-color cosine (cube) | 0.992 | **0.262** | <0.95 ✓ |
| L_align (final) | — | 0.206 | stable |
| L_analogy (final) | — | 0.011 | preserved |

### Wave 1 Remaining Gaps

**L_analogy Train/Val Gap:** 20.4x (val/train) — anti-shortcut scheduler DID NOT reduce gap (was 21x).

**Full-Corpus Retrieval:** Replaced batch-64 E3 evaluation with 5075-candidate corpus:

| Metric | Batch-64 | Full-Corpus (5075) |
|--------|----------|-------------------|
| Top-1 | 24.8% | **3.0%** |
| Top-3 | 44.2% | **4.6%** |
| Top-10 | — | **8.2%** |

Batch-64 overestimates analogy by 8x. Full-corpus Top-10=8.2% (target >50%).

### Per-Color Accuracy (fix model)

| Color | Accuracy | Main Confusion |
|-------|----------|---------------|
| Red | 80% | Orange |
| Blue | 80% | Orange |
| Green | 80% | Orange |
| Yellow | 20% | Blue |
| Purple | 80% | Orange |
| Orange | 40% | Red, Blue |

### Cross-Material Robustness (cube)

Matte 100%, Shiny 100%, Metallic 83%, Glass 83%.

---

## Manifold Structure Analysis (NEW — June 3)

### Critical Finding: Extreme Dimensionality Collapse

**256-dim manifold uses only ~4-6 effective dimensions.**

```
PCA spectrum (360 symbol embeddings):
  Dim 1: 29.9% variance
  Dim 2: 25.8%
  Dim 3: 22.0%
  Dim 4: 18.9%
  Dim 5:  2.4%  ← cliff
  Dim 6:  0.9%
  ...
  Dim 7+: <0.01% each

50% variance: 2 dims | 90%: 4 dims | 95%: 4 dims
Participation ratio: 6.0 effective dims
```

### Attribute Variance Distribution

| Attribute | Variance | Notes |
|-----------|----------|-------|
| Color | **62.2%** | Dominates (was 0% before fix — reversal!) |
| Size | 13.4% | Moderate |
| Shape | **4.0%** | Shape almost lost! |
| Material | 0.3% | Completely lost |

### Where the Collapse Happens

```
Per-category (64-dim)   ✓  Perfect disentanglement, same-shape = same OBJ emb
        ↓
MLP (384→512→256)       ✓  Balanced category weights (20% each), eff rank 60+
        ↓
MLP output (pre-proj)   ✓  Color direction consistent across shapes (cos=0.976)
        ↓
ManifoldProjection      ⚠  SpectralNorm + LayerNorm + L2 normalize
  (Linear 256→256)
        ↓
Final manifold output   ✗  4-6 effective dims, non-orthogonal attribute dirs
```

**The ManifoldProjection layer is where dimensionality collapses.** SpectralNorm limits singular values of the linear layer → if input has low effective rank, the projection can't "expand" into unused dimensions → LayerNorm + L2 normalize further compress.

### Color Direction Consistency (Across Shapes)

| Color pair | Cross-shape cos | Status |
|------------|----------------|--------|
| blue→yellow | 0.894 | ✓ Consistent |
| red→orange | 0.849 | ✓ |
| red→blue | 0.601 | ~ Marginal |
| red→green | 0.452 | ✗ Inconsistent |
| green→purple | 0.323 | ✗ Broken |

Color directions are NOT a true vector field — analogy arithmetic fails when direction varies by shape.

### Neighbor Purity (k=10)

| Attribute | Purity | × Chance |
|-----------|--------|----------|
| Color | 84.3% | 5.1x |
| Shape | 44.8% | 2.2x |
| Pyramid color | **25%** | near random |

### Why Attributes Can't Be Orthogonal

Three-layer diagnosis:

1. **Architecture**: MLP freely mixes categories. Per-cat embeddings are disentangled but the fully-connected projection combines them arbitrarily. No structural constraint preserves the direct sum M ≈ A_color ⊕ A_shape ⊕ ...

2. **Geometry**: Unit sphere normalization adds minor coupling (pre-norm ratio=0.996), but the main issue is that with only 4-6 effective dims, there's no "room" for orthogonal subspaces. Need ~18 orthogonal dimensions for all attributes.

3. **Loss**: L_disentangle only touches per-category embeddings (already good). No loss enforces subspace structure on the full output. L_align/L_rel/L_analogy work on collapsed space without penalty.

**Hypothesis is NOT wrong — architecture provides zero mechanism to realize it.**

---

## Key Files

```
sam/
├── encoders/
│   ├── visual.py              — ViT-Tiny (freeze L0-7)
│   └── symbol.py              — Per-cat embed → MLP (512 hidden) → 256
├── manifold/
│   └── projection.py          — SpectralNorm + LayerNorm + L2 (SUSPECT)
├── losses/
│   ├── color_contrastive.py   — Fixed: same-color-diff-shape positive
│   ├── symbol_color.py        — NEW: penalizes color collapse in full output
│   └── perturb.py             — Attribute perturbation regularization
├── data/
│   └── balanced_sampler.py    — Color-balanced batch sampling
scripts/
├── p2_wave1_train.py          — Wave 1 training (3-phase, 30 epochs)
├── full_evaluation.py         — G1-G5 comprehensive eval suite
├── wave1_completion.py        — Analogy gap + full-corpus retrieval
├── analyze_manifold.py        — A1-A7 manifold structure analysis
├── analyze_manifold_deep.py   — H1-H3 deep manifold diagnosis (partial)
├── diagnose_color_root_cause.py    — Root cause investigation
├── diagnose_shape_color_interaction.py — Shape-color interaction test
└── test_challenge_images.py   — Real image inference
docs/report/
├── phase2_wave1_color_fix_report.md  — Complete fix report
└── research_report.md                — Phase 1 research report
```

## Model Checkpoints

| Path | Phase | Epoch | Notes |
|------|-------|-------|-------|
| `outputs/p3/checkpoint_best.pt` | P3 (4-loss) | 35 | Pre-fix baseline |
| `outputs/p2_wave1/checkpoint_best.pt` | Wave 1 (old) | ~20 | Buggy color loss |
| `outputs/p2_wave1_fix/checkpoint_epoch030.pt` | Wave 1 (fixed) | 30 | **CURRENT BEST** |

## Next Session Priority

**P0: Diagnose ManifoldProjection dimensionality collapse**
- Analyze pre-proj vs post-proj PCA spectrum
- Test if SpectralNorm suppresses dimension diversity
- Try removing SpectralNorm and re-measuring effective dims

**P1: Restore shape representation**
- Color dominates 62% of variance, shape only 4%
- Need balanced attribute subspace allocation

**P2: Build structural direct-sum constraint**
- Replace or augment MLP with per-category output heads
- Add loss on full output that enforces subspace orthogonality

## Environment

- Conda env: `tct` (Python 3.12, PyTorch 2.12.0+cu132)
- GPU: RTX 4060 Laptop 8GB
- ViT weights: ModelScope `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`
- Use `conda run -n tct python` (NOT `conda activate`)
