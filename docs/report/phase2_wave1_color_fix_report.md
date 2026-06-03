# Phase 2 Wave 1: Color Blindness Fix — Completion Report

**Date:** 2026-06-03 | **Status:** COMPLETE ✓

---

## 1. Problem Statement

The SAM model could not distinguish colors. For any rendered shape, all 6 colors mapped to
nearly identical manifold embeddings (cosine > 0.99). Cross-modal color retrieval accuracy
was at chance level (17%), with each shape collapsing to a single "default color."

## 2. Previous (Incorrect) Hypothesis

The initial hypothesis was that the ViT 192→256 projection layer compressed color information.
This was **wrong**. Tests showed:
- ViT raw CLS features separate colors at 100% accuracy
- Color survives through VisualEncoder.projection (100% → 98.3% on solid squares)
- The visual pipeline preserves color; the problem was elsewhere

## 3. Root Cause (Found via Systematic Diagnosis)

**Primary: Symbol encoder MLP bottleneck (384→128→256) destroys color information.**

The symbol encoder concatenates per-category embeddings (6 categories × 64-dim = 384-dim),
then compresses through MLP: 384 → 128 → 256. The 128-dim hidden layer forces 3:1 compression.
During training, L_rel and L_analogy reward shape/relation preservation, while no loss term
directly protects color in the full output. The MLP learns to discard color.

**Evidence:**
- Per-category embeddings (64-dim) are well disentangled: cross-color cosine ~0.09
- After MLP, same-shape-diff-color symbol embeddings collapse: cosine **0.998**
- L_disentangle only regularizes per-category embeddings (already fine), not the MLP output

**Secondary: ColorContrastiveLoss had inverted semantics.**
- Defined same-shape-diff-color as POSITIVE pairs → pulled them together
- Loss preferred color collapse (0.693) over separation (1.651)
- Gradient confirmed: pushed same-shape-diff-color TOWARD each other

## 4. Fixes Applied

### 4.1 Widened Symbol Encoder MLP (sam/encoders/symbol.py)
- hidden_dim: 128 → 512
- Removes 3:1 compression bottleneck (384 → 512 → 256)
- +222K parameters (77K → 299K), 174/179 P3 weights preserved on load

### 4.2 Fixed ColorContrastiveLoss (sam/losses/color_contrastive.py)
- Inverted positive mask: `same_shape & diff_color` → `same_color & diff_shape`
- Now says "red cube and red sphere should be closer than red cube and blue cube"
- Collapse loss: 20.0 vs separation loss: 1.08

### 4.3 New SymbolColorSeparationLoss (sam/losses/symbol_color.py)
- Directly penalizes cosine > 0.9 for same-shape-diff-color symbol pairs
- Forces MLP to produce distinct outputs for different colors
- Weight: 0.2-0.3 across training phases

### 4.4 Training Script Updated (scripts/p2_wave1_train.py)
- Integrates SymbolColorSeparationLoss in both scene and analogy steps
- Manual state dict filtering for backward compatibility with P3 checkpoint

## 5. Results

### 5.1 Quantitative Results (30-epoch full training)

| Metric | P3 Baseline | After Fix | Change |
|--------|------------|-----------|--------|
| Cross-modal Top-1 (5×6 tests) | 16.7% | **63.3%** | +46.7pp |
| Cross-modal Top-3 (5×6 tests) | 53.3% | **80.0%** | +26.7pp |
| Cube color retrieval | 1/6 | **6/6** | Perfect |
| Symbol cross-color cosine (cube) | 0.992 | **0.262** | -73.7% |
| Symbol cross-color cosine (sphere) | 0.991 | **0.355** | -64.2% |
| Material robustness (avg) | N/A | **92%** | — |

### 5.2 Per-Color Accuracy

| Color | Accuracy | Main Confusion |
|-------|----------|---------------|
| Red | 80% (4/5) | Orange |
| Blue | 80% (4/5) | Orange |
| Green | 80% (4/5) | Orange |
| Yellow | 20% (1/5) | Blue |
| Purple | 80% (4/5) | Orange |
| Orange | 40% (2/5) | Red, Blue |

### 5.3 Per-Shape Breakdown

| Shape | Before | After |
|-------|--------|-------|
| Cube | 1/6 | **6/6** |
| Sphere | 1/6 | **4/6** |
| Cylinder | 1/6 | **4/6** |
| Cone | 1/6 | **4/6** |
| Pyramid | 1/6 | 1/6 |

Pyramid remains shape-dominated — visually similar to cone, color signal weaker.

### 5.4 Training Dynamics

| Loss | Epoch 1 | Epoch 30 | Trend |
|------|---------|----------|-------|
| L_align | 0.879 | 0.206 | ↓ converging |
| L_rel | 0.055 | 0.039 | ↓ stable |
| L_analogy | — | 0.011 | ↓ preserved |
| L_color (new) | 2.073 | 1.694 | ↓ improving |
| L_sym_color (new) | 0.031 | 0.004 | ↓ 8x decrease |

No NaN issues. No degradation to analogy or relation performance.

## 6. Challenge Image Test

| Image | Top Prediction | Color Correct? |
|-------|---------------|----------------|
| Red cube render | cone red (cos=0.860) | ✓ red |
| Blue square render | sphere yellow (cos=0.945) | ✗ (OOD shape) |
| Real object photo | pyramid yellow (cos=0.876) | ~ (OOD) |
| Scene photo | cone red (cos=0.871) | ✓ red dominant |

## 7. Remaining Issues

1. **Pyramid color collapse**: Symbol cross-color cosine still 0.966 for pyramid
   — pyramid/cone visual confusion overwhelms color signal
2. **Yellow-blue confusion**: Yellow accuracy only 20%, often confused with blue
3. **Orange ambiguity**: Orange confused with red (similar hues)
4. **L_color not fully converged**: Still at ~1.7, may benefit from more epochs

## 8. Files Changed

```
sam/encoders/symbol.py            — hidden_dim 128→512
sam/losses/color_contrastive.py   — inverted positive mask
sam/losses/symbol_color.py        — NEW: color separation loss
scripts/p2_wave1_train.py         — integrated new loss, compat loading
scripts/diagnose_color_root_cause.py     — NEW: root cause diagnosis
scripts/diagnose_shape_color_interaction.py — NEW: shape-color interaction test
scripts/full_evaluation.py        — NEW: comprehensive eval suite
scripts/test_challenge_images.py  — NEW: real image testing
docs/report/phase2_wave1_color_fix_report.md — THIS FILE
```

## 9. Conclusions

The color blindness was caused by the symbol encoder's MLP bottleneck destroying
color information, combined with a ColorContrastiveLoss that actively reinforced
color collapse. The visual pipeline (ViT + projection) was never the problem.

After fixes: cross-modal color accuracy improved from 16.7% to 63.3% (3.8x),
cube retrieval from 1/6 to 6/6 (perfect), and symbol embeddings now encode
meaningful color distinctions (cross-color cosine 0.26 vs 0.99 before).

**H1 (shared manifold enables compositional reasoning) remains supported.**
The fix demonstrates that both shape and color can coexist on the same manifold
when the architecture prevents information bottlenecks and the loss functions
correctly incentivize attribute-level organization.
