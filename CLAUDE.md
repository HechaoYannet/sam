# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

SAM (Shared Analogical Manifold) — a proof-of-concept for multimodal reasoning where vision and symbols coexist on the same continuous manifold. Reasoning = manifold dynamics, not token generation.

**Core hypothesis (H1):** If visual and symbolic representations share a disentangled continuous manifold, cross-modal analogical reasoning and compositional systematicity emerge as geometric properties — no large parameter count required.

**Hard constraint:** 8GB VRAM (RTX 4060 Laptop).

## Design Documents

- `dosc/design/architecture.md` — authoritative architecture spec. Read this first before any implementation work.
- `dosc/plan/initial_plan.md` — archived original MM-JEPA physics-prediction plan. Historical reference only.

## Key Architecture Constraints (from design doc)

- **Shared manifold:** 256-dim unit hypersphere. Both encoders output to the same space.
- **Visual encoder:** ViT-Tiny/16 (5.5M), ImageNet-1k pretrained, first 8 layers frozen.
- **Symbol encoder:** Per-category embedding + sum pooling → MLP → 256-dim (1.5M params). Structured tokens only (no free-form text in v1).
- **Four loss terms:** cross-modal alignment, relational consistency, analogy completion, disentanglement regularization.
- **Data:** Programmatic 2D rendering (shapes + attributes). Train/test split by attribute combination, not by scene — to test compositional systematicity.
- **Three eval experiments:** E1 compositional generalization, E2 cross-modal isomorphism (RSA), E3 zero-shot cross-modal analogy.
- **No text decoder in v1.** Interfaces reserved for future extension.
