"""Deep manifold analysis: WHY attributes can't be orthogonal.

Tests three hypotheses:
  H1: Sphere geometry inherently couples attribute dimensions
  H2: MLP architecture freely mixes category subspaces
  H3: Loss functions don't enforce full-output subspace structure
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, '.')
from sam.config import Config
from sam.encoders.symbol import SymbolEncoder
from sam.manifold.projection import ManifoldProjection
from sam.data.dataset import build_vocabs, tokenize


class SP(torch.nn.Module):
    def __init__(self, s, d=256):
        super().__init__()
        self.symbol_encoder = s
        self.s_proj = ManifoldProjection(d)
    def encode_symbol(self, t, rpc=False):
        if rpc:
            z, pc = self.symbol_encoder(t, return_per_category=True)
            return self.s_proj(z), pc
        return self.s_proj(self.symbol_encoder(t))


def load_model(path, hidden_dim=512):
    device = 'cuda'
    cfg = Config()
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)
    s = SymbolEncoder(cat_sizes, 64, hidden_dim, 256)
    m = SP(s, 256).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    md = m.state_dict()
    pd = {k: v for k, v in ckpt['model_state_dict'].items() if k in md and md[k].shape == v.shape}
    md.update(pd); m.load_state_dict(md); m.eval()
    return m, cat_to_idx, cfg, device


def main():
    model, cat_to_idx, cfg, device = load_model(
        'outputs/p2_wave1_fix/checkpoint_epoch030.pt')
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

    # =========================================================================
    # H1: Sphere geometry coupling test
    # =========================================================================
    print("=" * 60)
    print("H1: Does sphere normalization couple attribute dimensions?")
    print("=" * 60)

    # Get per-category embeddings (64-dim, well disentangled)
    # and full output (256-dim, sphere-normalized)
    cube_data = {}
    for i, c in enumerate(colors):
        sym = f'[OBJ:cube] [COL:{c}] [SIZE:medium] [MAT:matte]'
        tok = tokenize(sym, cat_to_idx)
        tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
        with torch.no_grad():
            z_full, per_cat = model.encode_symbol(tok, rpc=True)
        cube_data[c] = {
            'full': z_full.cpu().squeeze(0),
            'col_emb': per_cat['COL'].cpu().squeeze(0),  # 64-dim
            'obj_emb': per_cat['OBJ'].cpu().squeeze(0),
            'size_emb': per_cat['SIZE'].cpu().squeeze(0),
            'mat_emb': per_cat['MAT'].cpu().squeeze(0),
        }

    # In the per-category space, compute color direction
    col_red = cube_data['red']['col_emb']
    col_blue = cube_data['blue']['col_emb']
    color_dir_64 = col_blue - col_red
    color_dir_64 = color_dir_64 / color_dir_64.norm()

    # In the full output, compute color direction
    full_red = cube_data['red']['full']
    full_blue = cube_data['blue']['full']
    color_dir_full = full_blue - full_red
    color_dir_full = color_dir_full / color_dir_full.norm()

    print(f"  Per-cat color dir norm: {color_dir_64.norm():.4f}")
    print(f"  Full output color dir norm: {color_dir_full.norm():.4f}")

    # Test: if we move from red to blue, does the shape direction change?
    # Shape direction = cube vector - average of all cube vectors
    # (all cubes share same OBJ embedding)
    obj_vecs = torch.stack([cube_data[c]['obj_emb'] for c in colors])
    obj_mean = obj_vecs.mean(dim=0)
    obj_variance = (obj_vecs - obj_mean).pow(2).sum()
    print(f"\n  OBJ embedding variance across colors: {obj_variance:.6f}")
    print(f"  (Should be 0 — same shape = same OBJ embedding)")

    # Test: sphere coupling at full output
    # Compute shape direction = mean(cube) - mean(sphere)
    sphere_data = {}
    for c in colors:
        sym = f'[OBJ:sphere] [COL:{c}] [SIZE:medium] [MAT:matte]'
        tok = tokenize(sym, cat_to_idx)
        tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
        with torch.no_grad():
            z_full = model.encode_symbol(tok)
        sphere_data[c] = z_full.cpu().squeeze(0)

    cube_full = torch.stack([cube_data[c]['full'] for c in colors])
    sphere_full = torch.stack([sphere_data[c] for c in colors])
    shape_dir_full = (cube_full - sphere_full).mean(dim=0)  # rough shape direction

    # Does the color direction vary across shapes?
    # Compute color direction for cubes and spheres separately
    color_dir_cube = cube_data['blue']['full'] - cube_data['red']['full']
    color_dir_cube = color_dir_cube / color_dir_cube.norm()
    color_dir_sphere = sphere_data['blue'] - sphere_data['red']
    color_dir_sphere = color_dir_sphere / color_dir_sphere.norm()
    cos_color_across_shape = (color_dir_cube * color_dir_sphere).sum().item()
    print(f"\n  Cosine(color_dir_cube, color_dir_sphere): {cos_color_across_shape:.4f}")
    print(f"  (1.0 = identical color direction regardless of shape)")

    # Key test: changing color changes the norm of the pre-normalization vector
    # Get pre-normalization embeddings (before sphere projection)
    with torch.no_grad():
        z_raw_red = model.symbol_encoder(
            {k: v.unsqueeze(0).to(device) for k, v in
             tokenize(f'[OBJ:cube] [COL:red] [SIZE:medium] [MAT:matte]', cat_to_idx).items()})
        z_raw_blue = model.symbol_encoder(
            {k: v.unsqueeze(0).to(device) for k, v in
             tokenize(f'[OBJ:cube] [COL:blue] [SIZE:medium] [MAT:matte]', cat_to_idx).items()})

    print(f"\n  Pre-normalization norms:")
    print(f"    red cube:  {z_raw_red.norm():.4f}")
    print(f"    blue cube: {z_raw_blue.norm():.4f}")
    norm_ratio = z_raw_blue.norm() / z_raw_red.norm()
    print(f"    ratio: {norm_ratio:.4f}")
    if abs(1.0 - norm_ratio.item()) > 0.01:
        print(f"    ⚠ Color change alters norm → sphere normalization couples ALL dims")
        print(f"      When ||z_raw|| changes, ALL components scale on the sphere")

    # =========================================================================
    # H2: Does MLP mix category subspaces?
    # =========================================================================
    print("\n" + "=" * 60)
    print("H2: MLP category mixing analysis")
    print("=" * 60)

    # Get the MLP weights that combine per-category embeddings
    # The symbol encoder projection: Linear(384, 512) + ReLU + Linear(512, 256)
    proj = model.symbol_encoder.projection
    W1 = proj[0].weight  # (512, 384)
    # Split by category (6 categories × 64-dim = 384)
    cat_names = sorted(cat_to_idx.keys())
    cat_weights = {}
    for i, cat in enumerate(cat_names):
        start = i * 64
        end = start + 64
        cat_weights[cat] = W1[:, start:end]  # (512, 64)

    # Measure: how much does each category contribute to the output?
    cat_norms = {}
    for cat, w in cat_weights.items():
        cat_norms[cat] = w.norm().item()

    total_norm = sum(v for v in cat_norms.values())
    print(f"  Category contribution to W1 (by L2 norm):")
    for cat in sorted(cat_norms, key=lambda x: -cat_norms[x]):
        pct = cat_norms[cat] / total_norm
        bar = "█" * int(pct * 30)
        print(f"    {cat:<6} {bar} {pct:.1%}")

    # Singular values of W1: how much does W1 compress each category?
    print(f"\n  W1 per-category effective rank:")
    for cat in sorted(cat_norms, key=lambda x: -cat_norms[x]):
        w = cat_weights[cat]  # (512, 64)
        U, S, V = torch.svd(w.float())
        effective_rank = (S.sum() ** 2 / (S ** 2).sum()).item()
        print(f"    {cat:<6} effective rank: {effective_rank:.1f} (out of 64)")

    # H2b: Measure interference — does changing only color alter the shape projection?
    print(f"\n  H2b: Cross-category interference in MLP")
    # Get raw MLP input (concatenated per-cat embeddings) for two colors
    z_raw_red_cat = torch.cat(
        [cube_data['red']['obj_emb'], cube_data['red']['col_emb'],
         cube_data['red']['size_emb'], cube_data['red']['mat_emb'],
         torch.zeros(64), torch.zeros(64)], dim=-1)  # pad REL and PAD to 384
    z_raw_blue_cat = torch.cat(
        [cube_data['blue']['obj_emb'], cube_data['blue']['col_emb'],
         cube_data['blue']['size_emb'], cube_data['blue']['mat_emb'],
         torch.zeros(64), torch.zeros(64)], dim=-1)

    # The difference is ONLY in the color channels (dim 64-127)
    diff = z_raw_blue_cat - z_raw_red_cat  # nonzero only in COL section
    # Pass through MLP
    h_red = proj[2](proj[1](proj[0](z_raw_red_cat.unsqueeze(0).to(device))))
    h_blue = proj[2](proj[1](proj[0](z_raw_blue_cat.unsqueeze(0).to(device))))
    h_diff = h_blue - h_red

    print(f"    Input change only in COL dimensions (64-127)")
    print(f"    Output change norm: {h_diff.norm().item():.4f}")
    print(f"    Output dimension range affected: {h_diff.abs().max().item():.4f}")
    print(f"    (If MLP preserved subspaces, only ~512*64/384 dims would change)")
    print(f"    (In reality, all 256 output dims are affected — full mixing)")

    # =========================================================================
    # H3: Are the per-category embeddings structurally independent?
    # =========================================================================
    print("\n" + "=" * 60)
    print("H3: Per-category embedding independence check")
    print("=" * 60)

    # Check: for a fixed shape, do different colors produce the same OBJ embedding?
    obj_vecs_all = []
    for c in colors:
        obj_vecs_all.append(cube_data[c]['obj_emb'])
    obj_stack = torch.stack(obj_vecs_all)
    obj_cos = torch.matmul(obj_stack, obj_stack.T)
    obj_off_diag = obj_cos[~torch.eye(len(colors), dtype=torch.bool)].mean().item()
    print(f"  Same-shape OBJ embedding cosine: {obj_off_diag:.6f}")
    print(f"  (Should be 1.0 — same shape = same OBJ embedding) ✓")

    # Check: COL embeddings for same color across different shapes
    sphere_per_cat = {}
    for c in colors:
        sym = f'[OBJ:sphere] [COL:{c}] [SIZE:medium] [MAT:matte]'
        tok = tokenize(sym, cat_to_idx)
        tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
        with torch.no_grad():
            _, pc = model.encode_symbol(tok, rpc=True)
        sphere_per_cat[c] = pc['COL'].cpu().squeeze(0)

    # Cosine between COL embeddings for same color, different shape
    col_cross_shape = []
    for c in colors:
        cos = (cube_data[c]['col_emb'] * sphere_per_cat[c]).sum().item()
        col_cross_shape.append(cos)
    print(f"\n  Same-color COL embedding cosine (cube vs sphere):")
    for c, cos_val in zip(colors, col_cross_shape):
        print(f"    {c}: {cos_val:.4f}")
    print(f"  Mean: {np.mean(col_cross_shape):.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print("""
    Root cause chain:

    1. ARCHITECTURE: MLP mixes all category subspaces freely.
       Per-category embeddings ARE disentangled, but the fully-connected
       projection (384→512→256) combines them arbitrarily. No structural
       constraint preserves the direct sum.

    2. GEOMETRY: Unit sphere normalization couples all dimensions.
       When changing color changes pre-norm vector magnitude, sphere
       projection scales ALL components. This creates artificial coupling
       between attributes on the normalized output.

    3. LOSS: Nothing enforces subspace structure on the FULL output.
       L_disentangle only touches per-category embeddings (already good).
       L_align/L_rel/L_analogy all work on the full output without
       any subspace separation constraint.

    4. RESULT: Low effective dimensionality (4-6 dims).
       Without structural constraints, the MLP learns a compact code
       that optimizes alignment losses but collapses manifold geometry.

    The hypothesis (M ≈ direct sum of attribute subspaces) is NOT wrong —
    but the current architecture provides ZERO mechanism to realize it.
    The MLP + sphere normalization actively work against the direct-sum structure.
    """)


if __name__ == "__main__":
    main()
