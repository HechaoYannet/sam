"""Comprehensive manifold structure analysis.

Tools:
  A1: Effective dimensionality (PCA spectrum)
  A2: Attribute subspace variance explained
  A3: Color direction consistency across shapes
  A4: Neighbor purity (k-NN by shape, color)
  A5: Pairwise distance distribution & manifold utilization
  A6: Cross-attribute interference (non-orthogonality)
  A7: Per-dimension attribute sensitivity
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, '.')
from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.manifold.projection import ManifoldProjection
from sam.data.dataset import build_vocabs, tokenize
from sam.data.renderer import ShapeRenderer
from torchvision import transforms
from tqdm import tqdm


class SP(torch.nn.Module):
    def __init__(self, v, s, d=256):
        super().__init__()
        self.visual_encoder = v; self.symbol_encoder = s
        self.v_proj = ManifoldProjection(d); self.s_proj = ManifoldProjection(d)
    def encode_visual(self, x):
        return self.v_proj(self.visual_encoder(x))
    def encode_symbol(self, t, rpc=False):
        if rpc:
            z, pc = self.symbol_encoder(t, return_per_category=True)
            return self.s_proj(z), pc
        return self.s_proj(self.symbol_encoder(t))


def load_model(path, hidden_dim=512):
    device = 'cuda'
    cfg = Config(); cfg.model.vit_pretrained = False
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)
    v = VisualEncoder(cfg.model, 256)
    s = SymbolEncoder(cat_sizes, 64, hidden_dim, 256)
    m = SP(v, s, 256).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    md = m.state_dict()
    pd = {k: v for k, v in ckpt['model_state_dict'].items() if k in md and md[k].shape == v.shape}
    md.update(pd); m.load_state_dict(md); m.eval()
    return m, cat_to_idx, cfg, device


def collect_embeddings(model, cat_to_idx, device):
    """Collect symbol embeddings for all attribute combos of cube."""
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']

    data = []  # list of (embedding, shape_idx, color_idx, size_idx, mat_idx)
    for si, shape in enumerate(shapes):
        for ci, color in enumerate(colors):
            for szi, size in enumerate(['small', 'medium', 'large']):
                for mi, mat in enumerate(['matte', 'shiny', 'metallic', 'glass']):
                    sym = f'[OBJ:{shape}] [COL:{color}] [SIZE:{size}] [MAT:{mat}]'
                    tok = tokenize(sym, cat_to_idx)
                    tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
                    with torch.no_grad():
                        z, per_cat = model.encode_symbol(tok, rpc=True)
                    data.append({
                        'z': z.cpu().squeeze(0),
                        'shape': si, 'color': ci, 'size': szi, 'mat': mi,
                        'shape_name': shape, 'color_name': color,
                        'per_cat': {k: v.cpu().squeeze(0) for k, v in per_cat.items()},
                    })

    Z = torch.stack([d['z'] for d in data])  # (N, 256)
    return data, Z


def analyze_dimensions(Z):
    """A1: PCA-based effective dimensionality."""
    print("=" * 60)
    print("A1: Effective Dimensionality (PCA)")
    print("=" * 60)

    Z_centered = Z - Z.mean(dim=0)
    U, S, V = torch.pca_lowrank(Z_centered, q=min(100, Z.shape[0]))
    variance = (S ** 2) / (S ** 2).sum()

    cumsum = torch.cumsum(variance, dim=0)
    d50 = (cumsum > 0.5).nonzero()[0][0].item() + 1
    d90 = (cumsum > 0.9).nonzero()[0][0].item() + 1
    d95 = (cumsum > 0.95).nonzero()[0][0].item() + 1

    print(f"  Total dimensions: {Z.shape[1]}")
    print(f"  Dims for 50% variance: {d50} ({d50/Z.shape[1]:.1%})")
    print(f"  Dims for 90% variance: {d90} ({d90/Z.shape[1]:.1%})")
    print(f"  Dims for 95% variance: {d95} ({d95/Z.shape[1]:.1%})")
    print(f"  Top-10 singular values: {[f'{v:.4f}' for v in variance[:10].tolist()]}")

    # Participation ratio (effective dimensionality)
    pr = S.sum() ** 2 / (S ** 2).sum()
    print(f"  Participation ratio: {pr.item():.1f} (effective dims)")

    return {'d50': d50, 'd90': d90, 'd95': d95, 'pr': pr.item()}


def analyze_attribute_variance(data, Z):
    """A2: How much variance does each attribute explain?"""
    print("\n" + "=" * 60)
    print("A2: Attribute Subspace Variance Explained")
    print("=" * 60)

    n = Z.shape[0]
    total_var = ((Z - Z.mean(dim=0)) ** 2).sum()

    for attr_name, attr_key in [('Shape', 'shape'), ('Color', 'color'),
                                  ('Size', 'size'), ('Material', 'mat')]:
        # Group by attribute value
        groups = defaultdict(list)
        for i, d in enumerate(data):
            groups[d[attr_key]].append(i)

        # Between-group variance
        between_var = 0.0
        grand_mean = Z.mean(dim=0)
        for val, indices in groups.items():
            group_mean = Z[indices].mean(dim=0)
            between_var += len(indices) * ((group_mean - grand_mean) ** 2).sum()

        explained = between_var / total_var
        bar = "█" * int(explained * 50) + "░" * (50 - int(explained * 50))
        print(f"  {attr_name:<12} {bar} {explained:.1%}")


def analyze_color_directions(data, Z):
    """A3: Color direction consistency across shapes."""
    print("\n" + "=" * 60)
    print("A3: Color Direction Consistency Across Shapes")
    print("=" * 60)

    shapes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

    # For each color pair, compute the direction vector for each shape
    # Then measure cosine between the same color-pair direction across shapes
    color_pairs = [('red', 'blue'), ('red', 'green'), ('blue', 'yellow'),
                   ('green', 'purple'), ('red', 'orange')]

    for c1, c2 in color_pairs:
        # Get direction vector for each shape
        dirs = {}
        for shape in shapes:
            z1 = None; z2 = None
            for d in data:
                if d['shape_name'] == shape and d['color_name'] == c1 and d['size'] == 1 and d['mat'] == 1:
                    z1 = d['z']
                if d['shape_name'] == shape and d['color_name'] == c2 and d['size'] == 1 and d['mat'] == 1:
                    z2 = d['z']
            if z1 is not None and z2 is not None:
                dir_vec = z2 - z1
                dirs[shape] = dir_vec / dir_vec.norm()

        # Cross-shape cosine between direction vectors
        cos_values = []
        shape_list = sorted(dirs.keys())
        for i, s1 in enumerate(shape_list):
            for s2 in shape_list[i + 1:]:
                cos = (dirs[s1] * dirs[s2]).sum().item()
                cos_values.append(cos)

        mean_cos = np.mean(cos_values)
        std_cos = np.std(cos_values)
        status = "✓" if mean_cos > 0.5 else "✗"
        print(f"  {c1}→{c2}: mean_cos={mean_cos:.3f}±{std_cos:.3f} {status}")


def analyze_neighbor_purity(data, Z):
    """A4: k-NN purity by shape and color."""
    print("\n" + "=" * 60)
    print("A4: Neighbor Purity (k=10)")
    print("=" * 60)

    Z_norm = Z / Z.norm(dim=1, keepdim=True)
    sim = torch.matmul(Z_norm, Z_norm.T)  # (N, N)
    sim.fill_diagonal_(-1)  # exclude self
    _, knn_idx = sim.topk(10, dim=1)

    # Shape purity: fraction of 10-NN with same shape
    shape_purity = []
    color_purity = []
    for i, d in enumerate(data):
        neighbors = knn_idx[i]
        same_shape = sum(1 for j in neighbors if data[j]['shape'] == d['shape'])
        same_color = sum(1 for j in neighbors if data[j]['color'] == d['color'])
        shape_purity.append(same_shape / 10)
        color_purity.append(same_color / 10)

    shape_p = np.mean(shape_purity)
    color_p = np.mean(color_purity)
    chance_shape = 1 / 5  # 5 shapes
    chance_color = 1 / 6  # 6 colors

    print(f"  Shape purity: {shape_p:.1%} (chance={chance_shape:.0%}, {shape_p/chance_shape:.1f}x above chance)")
    print(f"  Color purity: {color_p:.1%} (chance={chance_color:.0%}, {color_p/chance_color:.1f}x above chance)")

    # Breakdown by shape
    print(f"\n  Per-shape neighbor color purity:")
    for si, shape in enumerate(['cube', 'sphere', 'cylinder', 'cone', 'pyramid']):
        idxs = [i for i, d in enumerate(data) if d['shape'] == si]
        cp = np.mean([color_purity[i] for i in idxs])
        print(f"    {shape:<10} color purity: {cp:.1%}")


def analyze_distance_distribution(data, Z):
    """A5: Pairwise distance distribution."""
    print("\n" + "=" * 60)
    print("A5: Manifold Distance Distribution")
    print("=" * 60)

    Z_norm = Z / Z.norm(dim=1, keepdim=True)

    # Sample random pairs for efficiency
    n = Z.shape[0]
    rng = np.random.RandomState(42)
    n_pairs = 5000
    idx1 = rng.choice(n, n_pairs)
    idx2 = rng.choice(n, n_pairs)

    cos_sim = (Z_norm[idx1] * Z_norm[idx2]).sum(dim=1)
    # Euclidean distance on unit sphere
    euclidean = (2 - 2 * cos_sim).sqrt()

    print(f"  Cosine similarity: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}")
    print(f"  Euclidean distance: mean={euclidean.mean():.4f}, std={euclidean.std():.4f}")
    print(f"  Min cosine: {cos_sim.min():.4f}, Max cosine: {cos_sim.max():.4f}")

    # Same-shape vs cross-shape distances
    same_shape_dists = []
    cross_shape_dists = []
    for i in range(n_pairs):
        shape_i = data[idx1[i]]['shape']
        shape_j = data[idx2[i]]['shape']
        d = euclidean[i].item()
        if shape_i == shape_j:
            same_shape_dists.append(d)
        else:
            cross_shape_dists.append(d)

    print(f"\n  Same-shape distance:   mean={np.mean(same_shape_dists):.4f}")
    print(f"  Cross-shape distance:  mean={np.mean(cross_shape_dists):.4f}")
    ratio = np.mean(same_shape_dists) / np.mean(cross_shape_dists)
    print(f"  Ratio (same/cross):    {ratio:.3f} (<1 means clustering by shape)")

    # Same-color vs cross-color distances
    same_color_dists = []
    cross_color_dists = []
    for i in range(n_pairs):
        col_i = data[idx1[i]]['color']
        col_j = data[idx2[i]]['color']
        d = euclidean[i].item()
        if col_i == col_j:
            same_color_dists.append(d)
        else:
            cross_color_dists.append(d)

    print(f"\n  Same-color distance:   mean={np.mean(same_color_dists):.4f}")
    print(f"  Cross-color distance:  mean={np.mean(cross_color_dists):.4f}")
    ratio_c = np.mean(same_color_dists) / np.mean(cross_color_dists)
    print(f"  Ratio (same/cross):    {ratio_c:.3f}")


def analyze_cross_attribute_interference(data, Z):
    """A6: Does changing color also change shape direction?"""
    print("\n" + "=" * 60)
    print("A6: Cross-Attribute Interference")
    print("=" * 60)

    shapes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']
    # For each shape, get the "shape centroid"
    shape_centroids = {}
    for shape in shapes:
        zs = []
        for d in data:
            if d['shape_name'] == shape and d['size'] == 1 and d['mat'] == 1:
                zs.append(d['z'])
        shape_centroids[shape] = torch.stack(zs).mean(dim=0)

    # Color direction for each shape
    for shape in shapes:
        z_red = None; z_blue = None
        for d in data:
            if d['shape_name'] == shape and d['color_name'] == 'red' and d['size'] == 1 and d['mat'] == 1:
                z_red = d['z']
            if d['shape_name'] == shape and d['color_name'] == 'blue' and d['size'] == 1 and d['mat'] == 1:
                z_blue = d['z']
        if z_red is None or z_blue is None:
            continue

        color_dir = z_blue - z_red
        color_dir = color_dir / color_dir.norm()

        # Project shape centroid differences onto color direction
        for other_shape in shapes:
            if other_shape == shape:
                continue
            shape_diff = shape_centroids[other_shape] - shape_centroids[shape]
            shape_diff = shape_diff / max(shape_diff.norm(), 1e-8)
            interference = abs((color_dir * shape_diff).sum().item())
            if interference > 0.3:
                print(f"  {shape} color-dir ⊥ {other_shape}-dir: cos={interference:.3f} ⚠ HIGH")


def main():
    model, cat_to_idx, cfg, device = load_model('outputs/p2_wave1_fix/checkpoint_epoch030.pt')

    print("Collecting embeddings (360 combos × 5 shapes)...")
    data, Z = collect_embeddings(model, cat_to_idx, device)
    print(f"Collected {len(data)} embeddings: {Z.shape}")

    analyze_dimensions(Z)
    analyze_attribute_variance(data, Z)
    analyze_color_directions(data, Z)
    analyze_neighbor_purity(data, Z)
    analyze_distance_distribution(data, Z)
    analyze_cross_attribute_interference(data, Z)


if __name__ == "__main__":
    main()
