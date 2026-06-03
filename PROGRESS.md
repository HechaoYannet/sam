# SAM Project — Progress & Handoff

**Last updated:** 2026-06-03 01:55 | **Current phase:** Phase 2 Wave 1 (structural fixes)

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

## What's Broken (Priorities for Next Session)

### P0: Color Blindness — Root Cause Found

ViT is NOT the problem. Pretrained ViT-Tiny raw features separate colors well (cross-color cosine 0.79-0.86).

**Root cause**: SAM's 192→256 projection layer compresses color information. Shape/relation gradients dominate during training, color signal is treated as noise and averaged out.

**Fix (not yet implemented)**: 
1. **Widen projection**: 192 → 512 → 256 (instead of 192 → 256)
2. **Color auxiliary head**: Add a 6-class color classifier on ViT features, train jointly with L_aux
3. **HSV augmentation**: Add `transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)` to training pipeline

Recommended: implement all three together.

### P1: RGB Overfitting — Debunked

Tested with ±15 RGB perturbation — cosine scores unchanged. Model is NOT overfitting to specific RGB values. The projector bottleneck is the real issue.

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

| Metric | Phase 1 | Wave 1 | Target |
|--------|---------|--------|--------|
| Color accuracy (synth) | 1/6 | 2/6 | >3/6 |
| Blue recognition | #3 | #1 ✓ | — |
| Green bias (rows with green #1) | 6/6 | 4/6 | <3/6 |
| L_analogy train/val gap | 21x | ? | <5x |

## Next Steps (Priority Order)

1. **Fix color via projector widening + aux head + HSV aug** — most impactful remaining fix
2. **Re-run Wave 1 training** with the fix, verify color accuracy > 3/6
3. **Implement full-corpus retrieval** (scripts/p2_full_retrieval.py) 
4. **Measure L_analogy train/val gap** post-anti-shortcut
5. **Update CL AUDE.md** with learned best practices

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
