# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

SAM (Shared Analogical Manifold) — a proof-of-concept for multimodal reasoning where vision and symbols coexist on the same continuous manifold. Reasoning = manifold dynamics, not token generation.

**Core hypothesis (H1):** If visual and symbolic representations share a disentangled continuous manifold, cross-modal analogical reasoning and compositional systematicity emerge as geometric properties — no large parameter count required.

**Hard constraint:** 8GB VRAM (RTX 4060 Laptop).

## Environment

- **Conda env:** `tct` (Python 3.12, PyTorch 2.12.0+cu132, CUDA 13.2)
- **Activate:** `conda activate tct`
- **ViT pretrained weights:** downloaded from ModelScope to `~/.cache/sam/pretrained/`. Set `SAM_PRETRAINED_VIT` env var:
  ```
  $env:SAM_PRETRAINED_VIT = "$env:USERPROFILE\.cache\sam\pretrained\timm\vit_tiny_patch16_224___augreg_in21k_ft_in1k"
  ```
  If not set, defaults to HuggingFace Hub (blocked in China).
- **Download weights (one-time):** `python scripts/download_vit_weights.py`
- **Verify environment:** `python scripts/check_env.py`
- **Verify GPU:** `python scripts/verify_gpu.py`

## Common Commands

```powershell
# Generate the full dataset (~20k scenes)
python scripts/generate_data.py --output_dir data/sam_dataset

# Train (all phases: warmup → relation → analogy → finetune)
python scripts/train.py --data_dir data/sam_dataset --output_dir outputs/run1

# Resume from checkpoint
python scripts/train.py --data_dir data/sam_dataset --output_dir outputs/run1 --resume

# Resume from specific checkpoint
python scripts/train.py --data_dir data/sam_dataset --output_dir outputs/run1 --resume checkpoint_epoch030.pt

# Visualize training progress (in another terminal)
tensorboard --logdir outputs/run1/tensorboard

# AI training companion — monitor progress automatically
python scripts/monitor.py --output_dir outputs/run1           # continuous watch (60s interval)
python scripts/monitor.py --output_dir outputs/run1 --once    # one-shot report
python scripts/monitor.py --output_dir outputs/run1 --interval 30  # custom interval

# Quick smoke test (mini dataset)
python scripts/generate_data.py --output_dir data/test_mini --n_train 100 --n_val 20 --n_test_iid 20 --n_test_ood 20
python scripts/train.py --data_dir data/test_mini --output_dir outputs/test_run --epochs 5
```

## Design Documents

- `docs/design/architecture.md` — authoritative architecture spec. Read this first before any implementation work.
- `docs/plan/initial_plan.md` — archived original MM-JEPA physics-prediction plan. Historical reference only.

## Key Architecture Constraints (from design doc)

- **Shared manifold:** 256-dim unit hypersphere. Both encoders output to the same space.
- **Visual encoder:** ViT-Tiny/16 (5.5M), ImageNet-1k pretrained, first 8 layers frozen.
- **Symbol encoder:** Per-category embedding + sum pooling → MLP → 256-dim (1.5M params). Structured tokens only (no free-form text in v1).
- **Four loss terms:** cross-modal alignment, relational consistency, analogy completion, disentanglement regularization.
- **Data:** Programmatic 2D rendering (shapes + attributes). Train/test split by attribute combination, not by scene — to test compositional systematicity.
- **Three eval experiments:** E1 compositional generalization, E2 cross-modal isomorphism (RSA), E3 zero-shot cross-modal analogy.
- **No text decoder in v1.** Interfaces reserved for future extension.

## Code Architecture

```
sam/
├── config.py         — DataConfig, ModelConfig, LossConfig, TrainConfig (dataclasses)
├── data/
│   ├── renderer.py   — 2D shape renderer (matplotlib, 5 shapes × 6 colors × 3 sizes × 4 materials)
│   ├── scene_generator.py — Combinatorial split + scene/analogy generation
│   └── dataset.py    — Tokenizer + SingleObject/Scene/Analogy datasets
├── encoders/
│   ├── visual.py     — ViT-Tiny/16 → CLS token → projection (supports local weights via SAM_PRETRAINED_VIT)
│   └── symbol.py     — Per-category embedding → sum pool → MLP projection
├── manifold/
│   └── projection.py — SpectralNorm + LayerNorm → unit sphere normalization
├── losses/
│   ├── align.py      — L_align: cosine + InfoNCE contrastive
│   ├── relational.py — L_rel: cross-modal displacement vector MSE
│   ├── analogy.py    — L_analogy: vector arithmetic completion
│   └── disentangle.py — L_disentangle: cross-category orthogonality
├── trainer.py        — SAMPipeline + SAMTrainer (4-phase curriculum, grad accum, AMP)
│                       TensorBoard logging, checkpoint/resume, plateau detection
├── eval/             — Evaluation modules (E1/E2/E3, not yet implemented)
scripts/
├── generate_data.py      — Full dataset generation entry point
├── train.py              — Training entry point (with VRAM check)
├── download_vit_weights.py — ModelScope weight downloader
├── check_env.py          — Dependency/GPU version check
├── verify_gpu.py         — GPU forward pass + VRAM measurement
└── monitor.py            — AI training companion: convergence detection,
                            plateau warnings, anomaly alerts, progress reports
```

## Current Status

- **P0 (Infrastructure):** Complete.
  - 5.78M params, peak VRAM ~49 MB (forward pass)
  - ModelScope pretrained weights downloaded (21.8 MB)
  - TensorBoard logging, checkpoint/resume, AI monitor all wired in
- **P1 (Alignment):** Next phase. Single-object alignment training with L_align.

### Training output structure

```
outputs/run1/
├── tensorboard/              — TensorBoard event files
├── checkpoint_epochXXX.pt    — Periodic checkpoints (full state)
├── checkpoint_best.pt        — Best model by validation loss
├── checkpoint_latest.pt      — Latest checkpoint (for --resume auto)
├── run_config.json           — Run configuration for reproducibility
├── run_summary.json          — Final summary after training completes
├── training_log.jsonl        — Structured event log (checkpoints, alerts)
└── monitor_report.json       — Latest monitor analysis
```
