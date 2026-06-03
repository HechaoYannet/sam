"""
Diagnostic: root cause of attribute blindness & trade-off patterns.

Investigates:
1. Is MAT blindness a data issue (rendering) or model issue?
2. Does fixing one attribute always break another? (variance trade-off)
3. What does the ViT visual encoder actually encode for each attribute?

Usage: conda run -n tct python scripts/diagnose_attribute_tradeoff.py
"""
import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.manifold.projection import ManifoldProjection
from sam.data.dataset import build_vocabs, SceneDataset, tokenize, collate_fn


def compute_attr_separability(Z, labels, n_classes):
    """Linear separability: simple nearest-centroid classifier."""
    Z_c = Z - Z.mean(dim=0)
    centroids = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids.append(Z_c[mask].mean(dim=0))
        else:
            centroids.append(torch.zeros(Z.shape[1]))
    C = torch.stack(centroids)  # (K, D)
    # Assign each point to nearest centroid
    dists = torch.cdist(Z_c.unsqueeze(0), C.unsqueeze(0)).squeeze(0)  # (N, K)
    pred = dists.argmin(dim=1)
    acc = (pred == labels).float().mean().item()
    return acc


def analyze_visual_encoder_attr(model, loader, device):
    """Check what the ViT encoder (pre-projection) encodes for each attribute."""
    model.eval()
    Z_pre_proj = []  # before ManifoldProjection
    Z_post = []
    attr_labels = {'OBJ': [], 'COL': [], 'SIZE': [], 'MAT': []}

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            # Get ViT features BEFORE projection
            vit_feat = model.visual_encoder.vit.forward_features(batch["image"])[:, 0, :]
            Z_pre_proj.append(vit_feat.cpu())
            Z_post.append(model.encode_visual(batch["image"]).cpu())

            tokens = batch["tokens"]
            for attr in ['OBJ', 'COL', 'SIZE', 'MAT']:
                attr_labels[attr].extend(tokens[attr][:, 0].cpu().tolist())

    Z_vit = torch.cat(Z_pre_proj, dim=0)  # (N, 192) ViT features
    Z_proj = torch.cat(Z_post, dim=0)     # (N, 256) projected

    print("\n  ViT encoder (192-dim, pre-projection) attribute separability:")
    for attr in ['OBJ', 'COL', 'SIZE', 'MAT']:
        labels = torch.tensor(attr_labels[attr])
        acc = compute_attr_separability(Z_vit, labels, len(labels.unique()))
        print(f"    {attr}: linear probe accuracy = {acc:.1%}")

    print("\n  After projection (256-dim, post-ManifoldProjection):")
    for attr in ['OBJ', 'COL', 'SIZE', 'MAT']:
        labels = torch.tensor(attr_labels[attr])
        acc = compute_attr_separability(Z_proj, labels, len(labels.unique()))
        print(f"    {attr}: linear probe accuracy = {acc:.1%}")

    # Check: does the ViT differentiate materials at all?
    print("\n  Material discrimination in ViT space:")
    mat_labels = torch.tensor(attr_labels['MAT'])
    mat_Z = Z_vit
    for mat_val in mat_labels.unique():
        mask = mat_labels == mat_val
        print(f"    class {mat_val}: {mask.sum().item()} samples, "
              f"norm={mat_Z[mask].norm(dim=-1).mean():.2f}")


def analyze_variance_tradeoff(model, cat_to_idx, device):
    """Track how variance shifts between attributes across the projection layers."""
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']
    sizes = ['small', 'medium', 'large']
    mats = ['matte', 'shiny', 'metallic', 'glass']

    attr_labels = {'OBJ': [], 'COL': [], 'SIZE': [], 'MAT': []}
    all_tokens = {cat: [] for cat in sorted(cat_to_idx.keys())}

    for si, shape in enumerate(shapes):
        for ci, color in enumerate(colors):
            for szi, size in enumerate(sizes):
                for mi, mat in enumerate(mats):
                    sym = f'[OBJ:{shape}] [COL:{color}] [SIZE:{size}] [MAT:{mat}]'
                    tok = tokenize(sym, cat_to_idx)
                    for cat in all_tokens:
                        all_tokens[cat].append(tok[cat])
                    attr_labels['OBJ'].append(si)
                    attr_labels['COL'].append(ci)
                    attr_labels['SIZE'].append(szi)
                    attr_labels['MAT'].append(mi)

    batch_tok = {k: torch.stack(v, dim=0).to(device) for k, v in all_tokens.items()}

    model.eval()
    with torch.no_grad():
        z_pre, per_cat = model.symbol_encoder(batch_tok, return_per_category=True)
        z_post = model.s_proj(z_pre)
        z_post = F.normalize(z_post, dim=-1)

    Z = z_post.cpu()
    total_var = ((Z - Z.mean(dim=0)) ** 2).sum()

    print("\n  Per-attribute variance decomposition (post-projection):")
    for attr_name in ['OBJ', 'COL', 'SIZE', 'MAT']:
        labels_t = torch.tensor(attr_labels[attr_name])
        grand_mean = Z.mean(dim=0)
        between_var = 0.0
        for val in labels_t.unique():
            mask = labels_t == val
            group_mean = Z[mask].mean(dim=0)
            between_var += mask.sum().item() * ((group_mean - grand_mean) ** 2).sum()
        explained = (between_var / total_var).item()
        within_var = total_var - between_var
        print(f"    {attr_name}: between={explained:.2%}, within={1-explained:.2%}")

    # Check: does the orthogonal projection HELP or HURT per-category structure?
    if hasattr(model.symbol_encoder, '_cat_allocations'):
        allocs = model.symbol_encoder._cat_allocations
        print("\n  Per-category subspace effective rank (pre-proj vs post-proj):")
        offset = 0
        for cat in sorted(allocs.keys()):
            alloc = allocs[cat]
            # Pre-projection block
            block_pre = z_pre[:, offset:offset + alloc]
            U_pre, S_pre, _ = torch.pca_lowrank(
                block_pre - block_pre.mean(dim=0),
                q=min(50, block_pre.shape[0], block_pre.shape[1]) - 1)
            er_pre = (S_pre.sum()**2 / (S_pre**2).sum()).item()

            # Post-projection: this block gets mixed by orthogonal rotation
            # We can't isolate it post-proj — the dimensions are scrambled
            print(f"    {cat} (alloc={alloc}): pre-proj eff_rank={er_pre:.1f}")
            offset += alloc

        # How much does orthogonal projection mix categories?
        # Check: correlation between pre-proj block structure and post-proj
        W = model.s_proj.proj.weight.data  # orthogonal matrix (256, 256)
        if W.dim() == 3:
            W = W[0]  # parametrization stores as (1, 256, 256)
        eye = torch.eye(W.shape[0], device=W.device)
        print(f"\n  Orthogonal projection mixing analysis:")
        print(f"    W shape: {W.shape}, W^T W ≈ I: "
              f"{(W.T @ W - eye).abs().max().item():.2e}")

        # How much does each pre-proj block contribute to each post-proj dimension?
        offset = 0
        for cat in sorted(allocs.keys()):
            alloc = allocs[cat]
            # W[output_dim, input_dim] — columns for this block
            W_block = W[:, offset:offset + alloc]  # (256, alloc)
            # Contribution of this block to each output dim
            block_norm = W_block.norm(dim=1)  # (256,)
            # Average across output dimensions
            avg_contribution = block_norm.mean().item()
            print(f"    {cat} block → output: avg L2 contribution = {avg_contribution:.4f}")
            offset += alloc


def analyze_zero_sum_variance(model, cat_to_idx, device):
    """Test: if we artificially boost one attribute's signal, do others shrink?"""
    print("\n  [Zero-Sum Test] L2 hypersphere constraint => variance budget")

    # Get embeddings
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']

    var_data = []
    for shape in shapes[:2]:  # Just test with 2 shapes
        for color in colors[:2]:  # and 2 colors
            sym = f'[OBJ:{shape}] [COL:{color}] [SIZE:medium] [MAT:matte]'
            tok = tokenize(sym, cat_to_idx)
            tok_dev = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
            with torch.no_grad():
                z = model.encode_symbol(tok_dev)
            var_data.append({
                'shape': shape, 'color': color,
                'z': z.cpu().squeeze(0),
            })

    # Compute shape-direction variance and color-direction variance
    Z = torch.stack([d['z'] for d in var_data])
    Z_c = Z - Z.mean(dim=0)

    # Shape separation: distance between same-color different-shape pairs
    shape_var = 0
    color_var = 0
    n_pairs = 0
    for i in range(len(var_data)):
        for j in range(i+1, len(var_data)):
            same_shape = var_data[i]['shape'] == var_data[j]['shape']
            same_color = var_data[i]['color'] == var_data[j]['color']
            diff = (Z[i] - Z[j]).norm().item() ** 2
            if same_color and not same_shape:
                shape_var += diff
                n_pairs += 1
            if same_shape and not same_color:
                color_var += diff

    print(f"    Shape separation (same color, diff shape): {shape_var/max(n_pairs,1):.4f}")
    print(f"    Color separation (same shape, diff color): {color_var/max(n_pairs,1):.4f}")
    print(f"    Total embedding norm: {Z.norm(dim=-1).mean():.4f} (all are 1.0 on sphere)")


def main():
    device = 'cuda'
    cfg = Config()
    cfg.device = device
    cfg.model.vit_pretrained = False
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

    checkpoints = {
        "v5b": "outputs/test_orthogonal_v5b/checkpoint_epoch015.pt",
    }

    with open("data/sam_dataset/scenes_train_meta.json") as f:
        scenes = json.load(f)[:500]

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"{'='*60}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_state = ckpt['model_state_dict']

        # Build model
        visual = VisualEncoder(cfg.model, cfg.model.manifold_dim)
        symbol = SymbolEncoder(cat_sizes, cfg.model.symbol_embed_dim,
                              manifold_dim=cfg.model.manifold_dim)

        class Model(torch.nn.Module):
            def __init__(self, v, s):
                super().__init__()
                self.visual_encoder = v
                self.symbol_encoder = s
                self.s_proj = ManifoldProjection(cfg.model.manifold_dim)
            def encode_visual(self, x):
                return self.s_proj(self.visual_encoder(x))
            def encode_symbol(self, tokens):
                return self.s_proj(self.symbol_encoder(tokens))
            def encode_symbol_raw(self, tokens):
                return self.symbol_encoder(tokens)

        model = Model(visual, symbol).to(device)
        md = model.state_dict()
        pd = {k: v for k, v in ckpt_state.items()
              if k in md and md[k].shape == v.shape}
        md.update(pd)
        model.load_state_dict(md)
        model.eval()

        # 1. Visual encoder attribute encoding
        scene_ds = SceneDataset(scenes, cat_to_idx)
        scene_loader = DataLoader(scene_ds, batch_size=64, shuffle=False,
                                  num_workers=0, collate_fn=collate_fn)
        analyze_visual_encoder_attr(model, scene_loader, device)

        # 2. Variance tradeoff
        analyze_variance_tradeoff(model, cat_to_idx, device)

        # 3. Zero-sum test
        analyze_zero_sum_variance(model, cat_to_idx, device)


if __name__ == "__main__":
    main()
