# SAM Project — Progress & Handoff

**Last updated:** 2026-06-04 | **Current phase:** Phase 2 Wave 4 Implementation Complete → Wave 5: Full Training + Eval

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

### Key Insights from Wave 2

- **"Dimension collapse" is a red herring.** v6 (eff_rank=6.2) destroys P3 (eff_rank=6.0) on all task metrics. Effective rank ≠ task performance.
- **PCA variance ≠ information content.** MAT at 0.2% PCA variance achieves 43% classification (random=25%). Information is encoded as subtle angular differences, not dominant principal components.
- **L2 hypersphere zero-sum variance budget.** On S^255, per-dim std bounded at ~1/√256 ≈ 0.0625. One attribute's variance gain is another's loss.

---

## Phase 2 Wave 3 — Complete: Scalable Regularization (VICReg + Decoder)

### Problem

Per-attribute losses (color_contrastive, GramCV, etc.) don't scale to real-world scenarios with hundreds of attributes. v6 baseline, while architecturally sound, still showed color-blindness (COL PCA variance 0.02%).

### Solution

Two attribute-agnostic components on top of v6 architecture:

| Component | File | Role |
|-----------|------|------|
| VICReg Covariance | `sam/losses/vicreg.py` | Decorrelate manifold dimensions (no attribute knowledge needed) |
| Decoder Bottleneck | `sam/losses/vicreg.py` | Reconstruct all attributes from embedding → forces information preservation |

Variance loss is **disabled** — LayerNorm in per-category heads already controls within-block variance, and the L2 hypersphere bounds per-dim std at ~0.06.

### v7 vs v6 Results (epoch 60, comprehensive eval)

#### E1: Cross-Modal Retrieval

| Split | v6 | v7 | Δ |
|-------|-----|-----|---|
| test_ood | 50.0% | **62.4%** | **+12.4pp** |
| test_iid | 53.2% | **67.4%** | **+14.2pp** |
| val | 45.4% | **56.4%** | **+11.0pp** |

#### E2: Cross-Modal Isomorphism

| Split | v6 RSA ρ | v7 RSA ρ |
|-------|----------|----------|
| test_ood | 0.943 | 0.938 (unchanged) |
| test_iid | 0.956 | 0.958 (unchanged) |

#### E3: Analogy (Direct Cosine)

| Split | v6 | v7 |
|-------|-----|-----|
| test_ood | 0.950 | 0.948 (unchanged) |
| test_iid | 0.961 | 0.960 (unchanged) |

#### Attribute Balance (PCA Variance)

| Attribute | v6 | v7 | Change |
|-----------|-----|-----|--------|
| OBJ (shape) | 57.34% | 59.54% | +2.2pp |
| COL (color) | **0.02%** | **0.74%** | **×37** |
| SIZE | 42.54% | 39.18% | -3.4pp |
| MAT (material) | **0.02%** | **0.47%** | **×23.5** |

#### Decoder Accuracies (v7)

OBJ 68.4% | COL 65.9% | SIZE 73.0% | MAT 70.0% | REL 100.0% — all balanced, all well above random.

#### Manifold Health

| Metric | v6 | v7 |
|--------|-----|-----|
| Effective Rank | 6.2 | **7.5** |
| cross_COL_MAT cos | 0.999 | 0.975 |

### Wave 3 Conclusion

**v7 is the new baseline.** VICReg + Decoder: +12.4pp E1, 37× COL information, zero E2/E3 degradation, 53K extra params (0.9%).

Full report: `docs/report/phase2_wave3_report.md`

---

## Phase 2 Wave 4 — Next: Analogy Fix & Continuous Attributes

### P1: Structured Analogy Data

**Problem:** E3 scene retrieval = 0% across all checkpoints. Current analogies are random pairings — the displacement vector `img_b - img_a` is noise.

**Fix:** Generate analogies where A and B differ by exactly ONE attribute:
```
img_a: red cube + blue sphere
img_b: red cube + green sphere  ← only sphere color changes
```
This forces the model to learn that analogy = single-attribute transformation vector.

### P2: Continuous Attribute Space

Extend from discrete tokens to continuous-valued attributes:
- RGB color encoding (small MLP: RGB → embedding)
- Continuous size/position
- Verify manifold smoothness: continuous attribute change → smooth trajectory on S^255

### P3: E3 Evaluation Redesign

Replace exact-scene-retrieval (infeasible with large candidate sets) with:
- Per-attribute retrieval accuracy (predicted OBJ/COL/SIZE/MAT match truth?)
- Displacement vector cosine with ground truth (already measured at 0.95)
- Single-attribute-change analogy accuracy

---

## Phase 2 Wave 4 — Complete: Structured Analogy + Continuous Color (v8)

### Architecture Changes (13 commits)

| Component | File | Change |
|-----------|------|--------|
| ColorEncoder | `sam/encoders/color_encoder.py` (NEW) | RGB(3)→32→64 MLP, maps continuous color to embedding space |
| Config | `sam/config.py` | 6 new fields: use_continuous_color, color_hue_samples, structured_analogy, etc. |
| Renderer | `sam/data/renderer.py` | `render_single_object_rgb`, `render_scene_rgb`, color_rgb_override support |
| Scene Generator | `sam/data/scene_generator.py` | `generate_structured_analogies`, `_make_variant`, `_sample_hues` |
| SymbolEncoder | `sam/encoders/symbol.py` | ColorEncoder integration, dual-path (discrete/continuous) |
| Dataset | `sam/data/dataset.py` | color_rgb in tokenize, SceneDataset, AnalogyDataset, collate_fn |
| Decoder | `sam/losses/vicreg.py` | RGB regression head (sigmoid+MSE) parallel to discrete COL head |
| Trainer | `sam/trainer.py` | color_rgb passthrough in _scene_step, _analogy_step, validate() |
| Evaluation | `scripts/comprehensive_eval.py` | L1 unseen hue retrieval, L4 structured analogy per-attribute accuracy |

### Key Design Decisions

1. **Dual-path color encoding:** Discrete tokens pick 6 points in the COL subspace (67-dim); continuous RGB covers the full space. Both share the same COL head — architecture guarantees color is a property of the subspace, not of the token.
2. **Structured analogies:** Each base scene generates N variants with single-attribute changes. Displacement vector encodes exactly one attribute change — making analogy learning tractable (was: random noise).
3. **Progressive strategy:** Discrete colors for warmup/relation phases (epoch 1-40), continuous color at analogy phase (epoch 41+). Config flags control the transition.
4. **Decoder RGB regression:** `torch.sigmoid(Linear→3)` outputs (R,G,B) in [0,1], MSE loss. Auto-switches from CE (discrete) to MSE (continuous) based on presence of `color_rgb` in batch.

### Smoke Test Results (5 epochs, 200 scenes)

All loss terms active and converging:
- loss_align: 1.44 → 0.47
- loss_rel: 0.75 (epoch 5, relation phase start)
- loss_decoder: 0.89 → 0.77
- VICReg covariance: active (0.0001)
- Decoder accs above random: OBJ 45%, COL 38%, SIZE 47%, MAT 38%, REL 37%

### Commits

```
f73a72e fix(eval): add color_rgb to SAMPipeline.encode_symbol in eval script
40b03f2 feat(data): add --structured_analogy/--random_analogy flags
09310cb feat(eval): add v8 L1 unseen hue retrieval and L4 structured analogy eval
92397a4 fix(trainer): pass color_rgb in validate() to fix train/val path mismatch
e96e5fa feat(trainer): wire continuous color through training pipeline
0e41087 feat(decoder): add RGB regression head for continuous color
df6c8e9 feat(dataset): add continuous color support to tokenize and SceneDataset
b083914 fix(data): wire generate_all to structured analogies, full UUID
956200d feat(encoders): integrate ColorEncoder into SymbolEncoder dual-path
a530a45 feat(data): add structured analogy generation and hue sampling
71d18e9 feat(renderer): add RGB continuous color rendering support
30b9307 feat(config): add v8 continuous color and structured analogy fields
a55c571 feat(encoders): add ColorEncoder for continuous RGB→embedding
```

### Next: Wave 5 — Full Training + Evaluation

- Run full 60-epoch training with v8 on complete dataset (20k scenes)
- Evaluate L0-L4 against v7 baseline
- Tune color_rgb_weight if needed
- Fallback Route B: full continuous retrain if progressive strategy shows regression

---

## Architecture Decisions Record

1. **SpectralNorm → Orthogonal:** Training dynamics cause irreversible rank collapse. Orthogonal constraint ($W^T W = I$) is the necessary fix.
2. **Shared MLP → Per-category heads (direct sum):** Shared parameters allow strong-signal attributes to invade weak-signal representation space.
3. **Per-attribute losses → VICReg + Decoder:** Only scalable approach. Decoder bottleneck forces information preservation; VICReg covariance keeps dimensions decorrelated.
4. **VICReg variance disabled:** LayerNorm in per-category heads already controls within-block variance. Adding global variance creates an unwinnable conflict.
5. **"Dimension collapse" → "Attribute balance":** Effective rank ≠ information content. Focus on per-attribute decodability, not PCA spectrum.
6. **256-dim manifold retained:** Only ~10 dim needed for current data, but capacity kept for continuous space and future extension.

---

## Model Checkpoints

| Path | Architecture | Epoch | E1 OOD | E2 RSA | Notes |
|------|-------------|-------|--------|--------|-------|
| `outputs/p2_wave1_fix/checkpoint_epoch030.pt` | MLP+SN | 30 | 24% | 0.61 | Old arch baseline |
| `outputs/test_orthogonal_v5b/checkpoint_epoch015.pt` | Heads+Ortho+pcdr+attr | 15 | 76% | 0.83 | Per-attribute losses (removed) |
| `outputs/v6_baseline/checkpoint_epoch060.pt` | Heads+Ortho | 60 | 50% | 0.94 | Clean baseline, color-blind |
| `outputs/v7_vicreg_decoder/checkpoint_epoch060.pt` | Heads+Ortho+VICReg+Decoder | 60 | **62%** | 0.94 | **Current best** |

---

## Key Files

```
sam/
├── encoders/
│   ├── visual.py              — ViT-Tiny (freeze L0-7)
│   └── symbol.py              — Per-category heads + direct sum
├── manifold/
│   └── projection.py          — Orthogonal parametrization
├── losses/
│   ├── align.py               — L_align: cross-modal InfoNCE
│   ├── relational.py          — L_rel: displacement vector MSE
│   ├── analogy.py             — L_analogy: vector arithmetic
│   ├── disentangle.py         — L_disentangle: cross-category orthogonality
│   └── vicreg.py              — VICReg covariance + DecoderBottleneck (NEW)
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
├── download_vit_weights.py    — ModelScope weight downloader
├── monitor.py                 — AI training companion
├── check_env.py               — Environment verification
└── verify_gpu.py              — GPU forward pass + VRAM check

docs/
├── design/architecture.md     — Authoritative architecture spec
├── plan/initial_plan.md       — Original MM-JEPA plan (historical)
└── report/
    ├── phase2_wave2_report.md — Root cause analysis + architecture fix report
    └── phase2_wave3_report.md — VICReg + Decoder results (NEW)
```

## Environment

- Conda env: `tct` (Python 3.12, PyTorch 2.12.0+cu132)
- GPU: RTX 4060 Laptop 8GB
- ViT weights: ModelScope `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`
- Use `conda run -n tct python` (NOT `conda activate`)
- TensorBoard: `tensorboard --logdir <full_path_to_output_dir>/tensorboard`
