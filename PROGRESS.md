# SAM Project — Progress & Handoff

**Last updated:** 2026-06-03 | **Current phase:** Phase 2 Wave 2 Complete → Wave 3: Scalable Regularization

---

## Quick Start

```powershell
$env:PYTHONPATH = "E:\PythonProject\mm-jepa"
$env:SAM_PRETRAINED_VIT = "$env:USERPROFILE\.cache\sam\pretrained\timm\vit_tiny_patch16_224___augreg_in21k_ft_in1k"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"  # OpenMP workaround
conda run -n tct python scripts/<script>.py
```

---

## Phase 2 Wave 2 — Complete: Architecture Diagnosis & Fix

### Key Findings (2026-06-03 Comprehensive Evaluation)

**The "dimensionality collapse" narrative was misleading.** Effective rank ~10 is adequate for the task. The real issues were attribute imbalance and architectural flaws.

### Root Cause Chain (Confirmed)

```
1. Old shared MLP (384→512→256): freely mixes all categories
   → Strong-signal attribute (COL) invades weak-signal representation space
2. SpectralNorm: σ_max grows during training → W/σ_max compresses ALL directions
   → Positive feedback collapse (effective rank: 27 → 6)
3. No structural constraints on final output
   → Model finds minimal representation (~6-10 dim) sufficient for alignment loss
```

### Fixes Applied

| Fix | File | Effect |
|-----|------|--------|
| SpectralNorm → Orthogonal | `sam/manifold/projection.py` | Prevents rank collapse (all σ=1) |
| MLP → Per-category heads (direct sum) | `sam/encoders/symbol.py` | Architecture-level attribute separation |
| Removed per-attribute losses | deleted `color_contrastive.py`, `symbol_color.py`, `perturb.py`, `dimensionality.py` | Not scalable to real-world scenarios |

### Comprehensive Evaluation Results (v5b, 200 scenes / 100 analogies)

| Experiment | Target | P3 (MLP+SN) | v4 (Heads+Ortho+GramCV) | v5b (Heads+Ortho+pcdr+attr) |
|------------|--------|-------------|------------------------|----------------------------|
| E1 Top1 test_ood | >70% | 24% | 54% | **76%** ✅ |
| E1 IID-OOD Gap | <15% | +4% | +1% | **+1%** ✅ |
| E2 mean_cos | >0.5 | 0.73 | 0.79 | **0.87** ✅ |
| E2 RSA ρ | >0.3 | 0.61 | 0.65 | **0.83** ✅ |
| E3 direct cosine | — | 0.87 | 0.75 | **0.83** (vector arithmetic works) |
| E3 scene retrieval | >40% | 0% | 0% | **0%** ❌ (exact match infeasible) |

### Manifold Health (360-attribute grid)

| Metric | P3 Baseline | v5b |
|--------|-------------|-----|
| Eff Rank | 6.0 | 10.3 |
| OBJ variance | 4.0% (shape-blind!) | 39.7% |
| COL variance | 62.2% (color-dominated) | 20.7% |
| SIZE variance | 13.4% | 37.7% |
| MAT variance | 0.3% | 0.2% |
| Cross-category cos (OBJ-MAT) | 0.985 (identical!) | 0.731 |
| MAT linear separability | 38.6% | 43.2% (random=25%, info IS encoded) |

**Critical insight:** MAT classification is 43.2% (well above random) despite 0.2% PCA variance. Information is encoded as subtle angular differences, not dominant principal components. **PCA variance ≠ information content.**

### Cleanup Completed

Removed legacy files (wrong direction, incompatible with current architecture):
- `sam/losses/color_contrastive.py`, `symbol_color.py`, `perturb.py`, `dimensionality.py`
- `scripts/diagnose_dimensionality.py`, `validate_orthogonal_fix.py`, `test_orthogonal_train.py`

---

## Phase 2 Wave 3 — Current: Scalable Architecture & Analogy Fix

### P0: Scalable Regularization (replaces per-attribute losses)

**Problem:** Per-attribute losses don't scale to real-world scenarios (hundreds of attributes, nested sub-attributes, cross-domain properties).

**Approach:** VICReg-style global regularization + decoder bottleneck:
- Variance regularization: prevent dimension collapse (global, no attribute knowledge needed)
- Covariance regularization: decorrelate dimensions
- Decoder bottleneck: small MLP reconstructs all attributes from embedding → information-theoretically forces preservation

**Why this scales:** The loss doesn't need to know the list of attributes. The decoder learns to extract whatever information the embedding contains.

### P1: Analogy Generalization Fix

**Problem:** E3 train=62% vs test=14% (48pp gap) — model memorizes training analogies.

**Root cause:** `scene_generator.py` generates analogies by randomly pairing unrelated scenes (img_a and img_b have no systematic relationship). The vector `img_b - img_a` is essentially noise.

**Fix:** Generate STRUCTURED analogies where A and B share all but ONE attribute:
```
img_a: red cube + blue sphere (left_of)
img_b: red cube + green sphere (left_of)  ← only sphere color changes
sym_a → sym_b: change COL[sphere] from blue to green
```
This forces the model to learn that analogy = single-attribute transformation.

### P2: Continuous Attribute Space

Extend from discrete tokens to continuous-valued attributes:
- RGB color encoding (small MLP: RGB → embedding)
- Continuous size/position
- Verify manifold smoothness: continuous attribute change → smooth trajectory on S^{255}

### P3: E3 Evaluation Redesign

Replace "exact scene retrieval" (infeasible with 100+ candidates) with:
- Per-attribute retrieval accuracy (predicted OBJ/COL/SIZE/MAT match truth?)
- Displacement vector cosine with ground truth (already measured at 0.83)
- Single-attribute-change analogy accuracy

---

## Architecture Decisions Record

1. **SpectralNorm → Orthogonal:** Training dynamics cause irreversible rank collapse. Orthogonal constraint ($W^T W = I$) is the necessary fix.
2. **Shared MLP → Per-category heads (direct sum):** Shared parameters allow strong-signal attributes to invade weak-signal representation space. Direct sum is architecture-level prevention.
3. **Per-attribute losses → Universal regularization:** Not scalable. VICReg + decoder bottleneck is the path forward.
4. **"Dimension collapse" → "Attribute balance":** Effective rank ≠ information content. Focus on per-attribute decodability, not PCA spectrum.
5. **256-dim manifold retained:** Only ~10 dim needed for current data, but capacity kept for continuous space and future extension.

---

## Key Files

```
sam/
├── encoders/
│   ├── visual.py              — ViT-Tiny (freeze L0-7)
│   └── symbol.py              — Per-category heads + direct sum (v1.2)
├── manifold/
│   └── projection.py          — Orthogonal parametrization (v1.2 fix)
├── losses/
│   ├── align.py               — L_align: cross-modal InfoNCE
│   ├── relational.py          — L_rel: displacement vector MSE
│   ├── analogy.py             — L_analogy: vector arithmetic
│   └── disentangle.py         — L_disentangle: cross-category orthogonality
├── data/
│   ├── renderer.py            — 2D shape renderer
│   ├── scene_generator.py     — Combinatorial split + scene/analogy generation
│   └── dataset.py             — Scene/Analogy datasets
├── trainer.py                 — 4-phase curriculum, TensorBoard, checkpoint
└── config.py                  — DataConfig, ModelConfig, LossConfig, TrainConfig

scripts/
├── generate_data.py           — Full dataset generation
├── train.py                   — Training entry point
├── comprehensive_eval.py      — Full E1/E2/E3 + manifold health evaluation
├── diagnose_attribute_tradeoff.py — Per-attribute information analysis
├── download_vit_weights.py    — ModelScope weight downloader
├── monitor.py                 — AI training companion
├── check_env.py               — Environment verification
└── verify_gpu.py              — GPU forward pass + VRAM check
```

## Model Checkpoints

| Path | Architecture | Epoch | E1 OOD | E2 RSA | Notes |
|------|-------------|-------|--------|--------|-------|
| `outputs/p2_wave1_fix/checkpoint_epoch030.pt` | MLP+SN | 30 | 24% | 0.61 | Old arch baseline |
| `outputs/test_orthogonal_v4/checkpoint_epoch015.pt` | Heads+Ortho+GramCV | 15 | 54% | 0.65 | Global Gram CV |
| `outputs/test_orthogonal_v5b/checkpoint_epoch015.pt` | Heads+Ortho+pcdr+attr | 15 | 76% | 0.83 | **Best current** |

## Environment

- Conda env: `tct` (Python 3.12, PyTorch 2.12.0+cu132)
- GPU: RTX 4060 Laptop 8GB
- ViT weights: ModelScope `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`
- Use `conda run -n tct python` (NOT `conda activate`)
- TensorBoard: `tensorboard --logdir <full_path_to_output_dir>/tensorboard`
