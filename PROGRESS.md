# SAM Project — Progress & Handoff

**Last updated:** 2026-06-03 13:45 | **Current phase:** Phase 2 Wave 1 — COLOR BLINDNESS FIXED ✓

---

## Color Blindness — Root Cause FOUND & FIXED (2026-06-03)

**Previous hypothesis (projection bottleneck 192→256) was WRONG.**

### True Root Cause

The **symbol encoder's MLP bottleneck (384→128→256)** destroys color information:
- Per-category embeddings (64-dim) are well disentangled (cross-color cosine ~0.1)
- But after MLP compression, same-shape-diff-color symbol embeddings collapse to **cosine 0.998**
- The 128-dim hidden layer forces 3:1 compression; model preserves shape (rewarded by L_rel/L_analogy) and discards color (no direct loss on full output)

### Contributing Bug

`ColorContrastiveLoss` had inverted semantics: same-shape-diff-color = POSITIVE (pulled them together). Loss preferred color collapse (0.693) over separation (1.651).

### Fixes Applied (3 changes)

1. **Fixed `ColorContrastiveLoss`** — inverted mask: same-color-diff-shape = positive now. Loss: 20.0 for collapse, 1.08 for separation.
2. **Widened symbol encoder MLP** — hidden_dim 128→512. Removes bottleneck. (+222K params, 174/179 P3 params loaded).
3. **Added `SymbolColorSeparationLoss`** — penalizes cosine >0.9 for same-shape-diff-color symbol pairs. Forces MLP to preserve color.

### Results (30-epoch full training)

| Metric | Before Fix | After Fix | Target |
|--------|-----------|-----------|--------|
| Symbol cross-color cosine | 0.998 | **0.262** | <0.95 ✓ |
| Visual cross-color cosine | 0.979 | **0.383** | — |
| Cross-modal Top-1 | 20.0% | **63.3%** | >50% ✓ |
| Cross-modal Top-3 | 56.7% | **80.0%** | — |
| Visual color acc (cube) | 85% | **100%** | — |
| Visual color acc (sphere) | 23% | **100%** | — |
| Cube color retrieval | 2/6 | **6/6** | >3/6 ✓ |
| L_align (final) | — | 0.206 | stable |
| L_analogy (final) | — | 0.011 | preserved |
| Symbol encoder params | 77K | 299K | — |

---

## Quick Start

```powershell
conda activate tct
$env:PYTHONPATH = "E:\PythonProject\mm-jepa"
$env:SAM_PRETRAINED_VIT = "$env:USERPROFILE\.cache\sam\pretrained\timm\vit_tiny_patch16_224___augreg_in21k_ft_in1k"
```

## Git History (17 commits)

```
b09497d P5: research report and Phase 1 paper draft
18808a2 P4: evaluation complete — H1 SUPPORTED
86bfceb P3: implement L_disentangle
79a64fa P3: fix NaN divergence
cf29538 P2: relation phase training
217be9c P1: single-object cross-modal alignment
... (earlier P0 commits)
```

Latest: `6f7ed7f` P2 Wave1: color contrastive loss, balanced sampler, anti-shortcut training

## Model Checkpoints

| Path | Phase | Best Epoch |
|------|-------|------------|
| `outputs/p3/checkpoint_best.pt` | P3 (4-loss) | 35 |
| `outputs/p2_wave1/checkpoint_best.pt` | Wave 1 (color fix + anti-shortcut) | ~20 |

## What Works

- **Shape recognition cross-domain**: cube always separates from sphere/cylinder/pyramid/cone with positive cosine margin
- **Compositional generalization**: IID-OOD gap only 2.4%
- **Cross-modal isomorphism**: RSA ρ = 0.957
- **Zero-shot analogy**: OOD Top-1 80%, Top-3 98.6%
- **Anti-shortcut training**: 3-phase scheduler implemented (recovery → analogy push → joint)
- **Color contrastive loss**: implemented but insufficient (1/6 → 2/6 accuracy)

## What's Fixed (June 3)

### Color Blindness — FIXED ✓

Root cause: symbol encoder MLP bottleneck (384→128→256) destroyed color information.
3 fixes: widened MLP (128→512), fixed ColorContrastiveLoss (inverted semantics), new SymbolColorSeparationLoss.
Result: cross-color cosine 0.998→0.262, retrieval 20%→63%, cube retrieval 2/6→6/6.

Remaining issues:
- Pyramid shape still poor (1/6) — likely pyramid/cone visual confusion
- L_color stays at ~1.7 (contrastive loss hasn't fully converged)
- Visual encoder for pyramid/cone still shape-dominated

### P1: RGB Overfitting — Debunked

Tested with ±15 RGB perturbation — cosine scores unchanged. Model is NOT overfitting to specific RGB values.

### P2: Alignment Shortcut — Partially Fixed

AntiShortcutScheduler implemented. L_perturb implemented. Need to verify L_analogy train/val gap reduced to <5x.

### P3: Full-Corpus Retrieval — Not Yet Implemented

Task #26 is pending. Script needs to:
- Precompute all 1360 candidate symbol embeddings
- Run 500 OOD analogy queries against full corpus
- Replace batch-64 evaluation in E3

## Key Files

```
sam/
├── losses/
│   ├── color_contrastive.py   # NEW Wave1 — same-shape-diff-color InfoNCE
│   └── perturb.py             # NEW Wave1 — attribute perturbation regularization
├── data/
│   └── balanced_sampler.py    # NEW Wave1 — color-balanced batch sampling
scripts/
├── p2_wave1_train.py          # Wave 1 training (3-phase, runs)
├── test_colors_synth.py       # Synthetic color accuracy test
├── test_color_overfit.py      # RGB perturbation overfitting test → DEBUNKED
├── test_vit_colors.py         # ViT raw color discrimination → ViT is fine!
├── test_single.py             # Single-object photo inference
├── p4_eval.py                 # Full P4 evaluation (E1/E2/E3)
```

## Current Wave 1 Results

| Metric | Phase 1 | Wave 1 (old) | Wave 1 Fix (3ep) | Target |
|--------|---------|-------------|-------------------|--------|
| Color accuracy (synth) | 1/6 | 2/6 | TBD | >3/6 |
| Symbol cross-color cos | — | 0.998 | **0.928** | <0.95 |
| ColorContrastLoss pref | collapse | collapse | **separation** | — |

Full 30-epoch training in progress.

## Next Steps (Priority Order)

1. **Complete full Wave 1 training** (30 epochs) — in progress
2. **Re-run diagnosis** on trained model: cross-color cosine, color retrieval accuracy
3. **Implement full-corpus retrieval** (scripts/p2_full_retrieval.py)
4. **Measure L_analogy train/val gap** post-anti-shortcut
5. **Update CLAUDE.md** with learned best practices

## Design Docs

- `dosc/design/architecture.md` — Phase 1 architecture (complete)
- `dosc/design/architecture_phase2.md` — Phase 2 architecture (design only)
- `dosc/report/research_report.md` — Full research report
- `dosc/paper/phase1_paper.md` — Phase 1 paper draft

## Environment

- Conda env: `tct` (Python 3.12, PyTorch 2.12.0+cu132)
- GPU: RTX 4060 Laptop 8GB
- ViT weights: ModelScope `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`
- TensorBoard: `tensorboard --logdir outputs/<run>/tensorboard --port 6006`
- HF blocked in China; use ModelScope for downloads
