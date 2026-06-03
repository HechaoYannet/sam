"""
Comprehensive SAM evaluation script.

Tests ALL architecture requirements (E1, E2, E3) across ALL data splits
for representative checkpoints. Also computes manifold health diagnostics.

Usage:
    conda run -n tct python scripts/comprehensive_eval.py
    tensorboard --logdir outputs/comprehensive_eval/tensorboard
"""

import json, sys, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config, ModelConfig
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.manifold.projection import ManifoldProjection
from sam.data.dataset import build_vocabs, SceneDataset, AnalogyDataset, collate_fn
from sam.losses.align import AlignmentLoss


# ==============================================================================
# Old SymbolEncoder (for loading p2_wave1_fix checkpoint)
# ==============================================================================
class SymbolEncoderOld(torch.nn.Module):
    """Original SymbolEncoder with shared MLP projection (pre heads-refactor)."""

    def __init__(self, cat_sizes, embed_dim=64, hidden_dim=512, manifold_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim
        self.embeddings = torch.nn.ModuleDict()
        for cat, vocab_size in cat_sizes.items():
            self.embeddings[cat] = torch.nn.Embedding(
                vocab_size, embed_dim, padding_idx=vocab_size - 1)
        total_embed_dim = len(cat_sizes) * embed_dim
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(total_embed_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, manifold_dim),
            torch.nn.LayerNorm(manifold_dim),
        )

    def forward(self, tokens, return_per_category=False):
        per_cat_raw = {}
        for cat, emb_layer in self.embeddings.items():
            if cat in tokens:
                cat_indices = tokens[cat]
                cat_emb = emb_layer(cat_indices)
                if cat_emb.dim() == 3:
                    cat_emb = cat_emb.mean(dim=1)
                per_cat_raw[cat] = cat_emb
            else:
                pad_idx = emb_layer.weight.shape[0] - 1
                per_cat_raw[cat] = emb_layer.weight[pad_idx].unsqueeze(0)
        parts = [per_cat_raw[cat] for cat in sorted(per_cat_raw.keys())]
        combined = torch.cat(parts, dim=-1)
        z = self.projection(combined)
        if return_per_category:
            return z, {cat: emb for cat, emb in per_cat_raw.items()}
        return z


# ==============================================================================
# Model wrapper
# ==============================================================================
class SAMPipeline(torch.nn.Module):
    def __init__(self, visual_encoder, symbol_encoder, manifold_dim=256,
                 use_spectral_norm=False):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.symbol_encoder = symbol_encoder
        if use_spectral_norm:
            self.v_proj = OldManifoldProjection(manifold_dim)
            self.s_proj = OldManifoldProjection(manifold_dim)
        else:
            self.v_proj = ManifoldProjection(manifold_dim)
            self.s_proj = ManifoldProjection(manifold_dim)

    def encode_visual(self, x):
        return self.v_proj(self.visual_encoder(x))

    def encode_symbol(self, tokens, return_per_category=False, return_pre_proj=False):
        if return_per_category or return_pre_proj:
            z_pre, per_cat = self.symbol_encoder(tokens, return_per_category=True)
            z_post = self.s_proj(z_pre)
            result = [z_post]
            if return_per_category:
                result.append(per_cat)
            if return_pre_proj:
                result.append(z_pre)
            return tuple(result) if len(result) > 1 else result[0]
        return self.s_proj(self.symbol_encoder(tokens))


class OldManifoldProjection(torch.nn.Module):
    """ManifoldProjection with SpectralNorm (old version)."""
    def __init__(self, dim=256):
        super().__init__()
        self.proj = torch.nn.utils.parametrizations.spectral_norm(
            torch.nn.Linear(dim, dim))
        self.ln = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return F.normalize(self.ln(self.proj(x)), p=2, dim=-1)


# ==============================================================================
# Evaluation metrics
# ==============================================================================
def effective_rank(S):
    return (S.sum() ** 2 / (S ** 2).sum()).item()


def compute_pca(Z):
    Z_c = Z - Z.mean(dim=0)
    U, S, V = torch.pca_lowrank(Z_c, q=min(100, min(Z.shape)))
    variance = (S ** 2) / (S ** 2).sum()
    cumsum = torch.cumsum(variance, dim=0)
    d50 = (cumsum > 0.5).nonzero()[0][0].item() + 1
    d90 = (cumsum > 0.9).nonzero()[0][0].item() + 1
    er = effective_rank(S)
    return er, d50, d90, variance, S


def evaluate_e1_retrieval(model, loader, device):
    """E1: Cross-modal retrieval accuracy — Top-1, Top-3, Top-5, Recall@10."""
    model.eval()
    all_z_v, all_z_s = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            z_v = model.encode_visual(batch["image"])
            z_s = model.encode_symbol(batch["tokens"])
            all_z_v.append(F.normalize(z_v, dim=-1))
            all_z_s.append(F.normalize(z_s, dim=-1))

    Z_v = torch.cat(all_z_v, dim=0)  # (N, D)
    Z_s = torch.cat(all_z_s, dim=0)
    N = Z_v.shape[0]

    sim = torch.matmul(Z_v, Z_s.T)  # (N, N)
    _, pred = sim.topk(10, dim=1)
    target = torch.arange(N, device=device)

    top1 = (pred[:, 0] == target).float().mean().item()
    top3 = (pred[:, :3] == target.unsqueeze(1)).any(dim=1).float().mean().item()
    top5 = (pred[:, :5] == target.unsqueeze(1)).any(dim=1).float().mean().item()
    recall10 = (pred == target.unsqueeze(1)).any(dim=1).float().mean().item()

    return {"e1_top1": top1, "e1_top3": top3, "e1_top5": top5, "e1_recall10": recall10}


def evaluate_e2_isomorphism(model, loader, device):
    """E2: Cross-modal relation isomorphism — displacement vector cosine + RSA."""
    model.eval()
    all_d_v, all_d_s = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            z_v = model.encode_visual(batch["image"])
            z_s = model.encode_symbol(batch["tokens"])
            B = z_v.shape[0]
            if B >= 2:
                # All pairwise displacement vectors within batch
                for i in range(B):
                    for j in range(i + 1, B):
                        d_v = z_v[i] - z_v[j]
                        d_s = z_s[i] - z_s[j]
                        all_d_v.append(d_v.cpu())
                        all_d_s.append(d_s.cpu())

    if len(all_d_v) < 2:
        return {"e2_mean_cosine": 0, "e2_median_cosine": 0, "e2_rsa_rho": 0,
                "e2_rsa_rank": 0}

    D_v = torch.stack(all_d_v)  # (M, D)
    D_s = torch.stack(all_d_s)

    # Per-pair cosine
    cosines = (F.normalize(D_v, dim=-1) * F.normalize(D_s, dim=-1)).sum(dim=-1)
    mean_cos = cosines.mean().item()
    median_cos = cosines.median().item()

    # RSA: Representational Similarity Analysis
    # Build RDMs: correlation matrix of displacement vectors
    M = min(len(D_v), 500)  # Cap for memory
    if len(D_v) > M:
        idx = torch.randperm(len(D_v))[:M]
        D_v, D_s = D_v[idx], D_s[idx]

    rdm_v = torch.matmul(D_v, D_v.T)  # (M, M)
    rdm_s = torch.matmul(D_s, D_s.T)

    # Pearson correlation between upper triangles
    mask = torch.triu(torch.ones(M, M), diagonal=1).bool()
    rdm_v_flat = rdm_v[mask]
    rdm_s_flat = rdm_s[mask]

    rsa_rho = torch.corrcoef(torch.stack([rdm_v_flat, rdm_s_flat]))[0, 1].item()

    # Rank correlation
    from scipy.stats import spearmanr
    rsa_rank, _ = spearmanr(rdm_v_flat.numpy(), rdm_s_flat.numpy())

    return {"e2_mean_cosine": mean_cos, "e2_median_cosine": median_cos,
            "e2_rsa_rho": rsa_rho, "e2_rsa_rank": rsa_rank}


def evaluate_e3_analogy(model, analogy_loader, cat_to_idx, device):
    """E3: Zero-shot cross-modal analogy completion.

    Two evaluation modes:
    1. Direct cosine: cosine(E_v(Ib)-E_v(Ia)+E_s(Sa), E_s(Sb_true))
       — measures whether vector arithmetic works in embedding space
    2. Scene codebook retrieval: find nearest scene embedding to predicted
       — measures whether the exact scene can be retrieved
    """
    model.eval()

    # Build scene codebook: collect ALL sym_b embeddings (same order as query)
    cb_z_list = []
    n_total = 0
    with torch.no_grad():
        for batch in analogy_loader:
            batch = to_device(batch, device)
            z_sb = model.encode_symbol(batch["sym_b"])
            cb_z_list.append(F.normalize(z_sb, dim=-1).cpu())
            n_total += z_sb.shape[0]

    cb_z = torch.cat(cb_z_list, dim=0).to(device)  # (N_total, D)

    # Evaluate analogies
    direct_cosines = []
    top1 = top3 = top5 = 0
    global_idx = 0  # tracks position in codebook

    with torch.no_grad():
        for batch in tqdm(analogy_loader, desc="E3 analogy"):
            batch = to_device(batch, device)
            B = batch["img_a"].shape[0]

            z_va = model.encode_visual(batch["img_a"])
            z_vb = model.encode_visual(batch["img_b"])
            z_sa = model.encode_symbol(batch["sym_a"])
            z_sb_true = model.encode_symbol(batch["sym_b"])

            # Predict: z_sb_pred = E_v(Ib) - E_v(Ia) + E_s(Sa)
            z_pred = z_vb - z_va + z_sa
            z_pred = F.normalize(z_pred, dim=-1)

            # ---- Metric 1: Direct cosine similarity with ground truth ----
            cos_direct = (z_pred * z_sb_true).sum(dim=-1)
            direct_cosines.append(cos_direct.cpu())

            # ---- Metric 2: Codebook retrieval (leave-one-out) ----
            sim = torch.matmul(z_pred, cb_z.T)  # (B, N_total)

            for i in range(B):
                gt_idx = global_idx + i
                if gt_idx >= n_total:
                    continue

                # Mask out the ground truth for leave-one-out
                sim_i = sim[i].clone()
                sim_i[gt_idx] = -float('inf')

                _, pred = sim_i.topk(5)
                if pred[0] == gt_idx:
                    top1 += 1
                if gt_idx in pred[:3]:
                    top3 += 1
                if gt_idx in pred[:5]:
                    top5 += 1

            global_idx += B

    direct_cos = torch.cat(direct_cosines) if direct_cosines else torch.zeros(0)
    mean_direct_cos = direct_cos.mean().item() if len(direct_cos) > 0 else 0
    median_direct_cos = direct_cos.median().item() if len(direct_cos) > 0 else 0

    if n_total == 0:
        return {"e3_direct_cosine": 0, "e3_median_cosine": 0,
                "e3_retrieval_top1": 0, "e3_retrieval_top3": 0, "e3_retrieval_top5": 0}

    return {
        "e3_direct_cosine": mean_direct_cos,
        "e3_median_cosine": median_direct_cos,
        "e3_retrieval_top1": top1 / n_total,
        "e3_retrieval_top3": top3 / n_total,
        "e3_retrieval_top5": top5 / n_total,
    }


def compute_manifold_health(model, cat_to_idx, device):
    """Full manifold health: effective rank, per-attribute variance, disentanglement."""
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']
    sizes = ['small', 'medium', 'large']
    mats = ['matte', 'shiny', 'metallic', 'glass']

    from sam.data.dataset import tokenize

    # Build all 360 combinations as batch
    all_tokens = {cat: [] for cat in sorted(cat_to_idx.keys())}
    attr_labels = {'OBJ': [], 'COL': [], 'SIZE': [], 'MAT': []}
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
    Z_pre = z_pre.cpu()
    Z_c = Z - Z.mean(dim=0)

    # PCA of final representation
    U, S, V = torch.pca_lowrank(Z_c, q=min(100, Z.shape[0]))
    variance = (S ** 2) / (S ** 2).sum()
    cumsum = torch.cumsum(variance, dim=0)
    er = effective_rank(S)
    d50 = (cumsum > 0.5).nonzero()[0][0].item() + 1
    d90 = (cumsum > 0.9).nonzero()[0][0].item() + 1
    d95 = (cumsum > 0.95).nonzero()[0][0].item() + 1

    # Top singular values
    sv_dict = {f"pca_sv{i+1:02d}": variance[i].item()
               for i in range(min(10, len(variance)))}

    # Per-attribute variance explained
    total_var = ((Z - Z.mean(dim=0)) ** 2).sum()
    attr_var = {}
    for attr_name in ['OBJ', 'COL', 'SIZE', 'MAT']:
        labels_t = torch.tensor(attr_labels[attr_name])
        grand_mean = Z.mean(dim=0)
        between_var = 0.0
        for val in labels_t.unique():
            mask = labels_t == val
            group_mean = Z[mask].mean(dim=0)
            between_var += mask.sum().item() * ((group_mean - grand_mean) ** 2).sum()
        attr_var[f"attr_{attr_name}_var"] = (between_var / total_var).item()

    # Cross-category disentanglement (cosine between category direction vectors)
    disentangle = {}
    cat_directions = {}
    for attr_name in ['OBJ', 'COL', 'SIZE', 'MAT']:
        labels_t = torch.tensor(attr_labels[attr_name])
        directions = []
        for val in labels_t.unique():
            mask = labels_t == val
            if mask.sum() > 1:
                # Direction = mean embedding of this attribute value
                directions.append(Z[mask].mean(dim=0))
        if len(directions) >= 2:
            cat_directions[attr_name] = torch.stack(directions)

    # Cross-category orthogonality
    cat_names = sorted(cat_directions.keys())
    for i, c1 in enumerate(cat_names):
        for c2 in cat_names[i+1:]:
            cos = torch.matmul(
                F.normalize(cat_directions[c1], dim=-1),
                F.normalize(cat_directions[c2], dim=-1).T,
            )
            disentangle[f"cross_{c1}_{c2}_mean_abs_cos"] = cos.abs().mean().item()

    # Pre-projection per-category effective rank (only for new heads architecture)
    try:
        allocations = model.symbol_encoder._cat_allocations
        pca_per_cat = {}
        offset = 0
        for cat in sorted(allocations.keys()):
            alloc = allocations[cat]
            if alloc >= 8:
                block = Z_pre[:, offset:offset + alloc]  # (N, alloc)
                q_val = min(50, min(block.shape[0], block.shape[1]) - 1)
                if q_val >= 2:
                    Uc, Sc, Vc = torch.pca_lowrank(
                        block - block.mean(dim=0), q=q_val)
                    pca_per_cat[f"preproj_{cat}_eff_rank"] = effective_rank(Sc)
            offset += alloc
    except (AttributeError, KeyError):
        pca_per_cat = {}

    return {
        "eff_rank": er, "d50": d50, "d90": d90, "d95": d95,
        **sv_dict, **attr_var, **disentangle, **pca_per_cat,
    }


# ==============================================================================
# Helpers
# ==============================================================================
def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def detect_checkpoint_architecture(ckpt_path):
    """Detect if checkpoint uses old (projection) or new (heads) architecture."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    keys = list(ckpt['model_state_dict'].keys())
    has_projection = any('symbol_encoder.projection' in k for k in keys)
    has_heads = any('symbol_encoder.heads' in k for k in keys)
    return 'old' if has_projection else ('new' if has_heads else 'unknown')


def build_model_for_checkpoint(ckpt_path, cfg, cat_sizes, device):
    """Build appropriate model architecture for a given checkpoint."""
    arch_type = detect_checkpoint_architecture(ckpt_path)
    print(f"  Architecture: {arch_type}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_state = ckpt['model_state_dict']

    if arch_type == 'old':
        visual = VisualEncoder(cfg.model, cfg.model.manifold_dim)
        symbol = SymbolEncoderOld(cat_sizes, cfg.model.symbol_embed_dim)
        model = SAMPipeline(visual, symbol, cfg.model.manifold_dim,
                          use_spectral_norm=True).to(device)
        # Selective load
        model_dict = model.state_dict()
        matched = {k: v for k, v in ckpt_state.items()
                  if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"    Loaded {len(matched)}/{len(model_dict)} params")
    else:
        visual = VisualEncoder(cfg.model, cfg.model.manifold_dim)
        symbol = SymbolEncoder(cat_sizes, cfg.model.symbol_embed_dim,
                              manifold_dim=cfg.model.manifold_dim)
        model = SAMPipeline(visual, symbol, cfg.model.manifold_dim).to(device)
        model_dict = model.state_dict()
        matched = {k: v for k, v in ckpt_state.items()
                  if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"    Loaded {len(matched)}/{len(model_dict)} params")

    model.eval()
    return model


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/sam_dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/comprehensive_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_scenes", type=int, default=500,
                       help="Max scenes per split (limits eval time)")
    parser.add_argument("--max_analogies", type=int, default=200,
                       help="Max analogies per split")
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.model.vit_pretrained = False  # Use ModelScope weights

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    log_dir = output_dir / "tensorboard"
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Vocab
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

    # Load data
    splits = {
        "train": ("scenes_train_meta.json", "analogy_train_meta.json"),
        "val": ("scenes_val_meta.json", "analogy_val_meta.json"),
        "test_iid": ("scenes_test_iid_meta.json", "analogy_test_iid_meta.json"),
        "test_ood": ("scenes_test_ood_meta.json", "analogy_test_ood_meta.json"),
    }

    print("=" * 70)
    print("S A M   C O M P R E H E N S I V E   E V A L U A T I O N")
    print("=" * 70)

    # Checkpoints to evaluate
    checkpoints = {
        "v6_baseline_ep060": "outputs/v6_baseline/checkpoint_epoch060.pt",
        "P3_baseline_ep030": "outputs/p2_wave1_fix/checkpoint_epoch030.pt",
        "v5b_heads_pcdr_attr_ep015": "outputs/test_orthogonal_v5b/checkpoint_epoch015.pt",
    }

    all_results = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"\n[SKIP] {ckpt_name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"  Path: {ckpt_path}")
        print(f"{'=' * 70}")

        t0 = time.time()
        model = build_model_for_checkpoint(ckpt_path, cfg, cat_sizes, args.device)

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {n_params:,} total, {n_trainable:,} trainable")

        ckpt_results = {}

        # ---- Manifold Health (on full 360 combos) ----
        print("\n  [Manifold Health] Full 360-attribute grid...")
        health = compute_manifold_health(model, cat_to_idx, args.device)
        for k, v in health.items():
            ckpt_results[f"health/{k}"] = v
            if k in ("eff_rank", "d50", "d90", "d95"):
                print(f"    {k}: {v:.1f}")
        for k, v in health.items():
            if k.startswith("attr_"):
                print(f"    {k}: {v:.2%}")
            if k.startswith("cross_"):
                print(f"    {k}: {v:.4f}")
            if k.startswith("preproj_"):
                print(f"    {k}: {v:.1f}")

        # ---- Per-split evaluations ----
        for split_name, (scene_file, analogy_file) in splits.items():
            print(f"\n  --- {split_name.upper()} ---")

            with open(data_dir / scene_file) as f:
                scenes = json.load(f)
            if args.max_scenes and len(scenes) > args.max_scenes:
                scenes = scenes[:args.max_scenes]

            scene_ds = SceneDataset(scenes, cat_to_idx)
            scene_loader = DataLoader(
                scene_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=0, collate_fn=collate_fn,
            )

            # E1: Retrieval
            e1 = evaluate_e1_retrieval(model, scene_loader, args.device)
            for k, v in e1.items():
                ckpt_results[f"{split_name}/{k}"] = v
            print(f"    E1 Retrieval: Top1={e1['e1_top1']:.1%} Top3={e1['e1_top3']:.1%} "
                  f"Top5={e1['e1_top5']:.1%} R@10={e1['e1_recall10']:.1%}")

            # E2: Isomorphism
            e2 = evaluate_e2_isomorphism(model, scene_loader, args.device)
            for k, v in e2.items():
                ckpt_results[f"{split_name}/{k}"] = v
            print(f"    E2 Isomorphism: mean_cos={e2['e2_mean_cosine']:.4f} "
                  f"RSA_rho={e2['e2_rsa_rho']:.4f} RSA_rank={e2['e2_rsa_rank']:.4f}")

            # E3: Analogy (if analogy data available)
            analogy_path = data_dir / analogy_file
            if analogy_path.exists():
                with open(analogy_path) as f:
                    analogies = json.load(f)
                if args.max_analogies and len(analogies) > args.max_analogies:
                    analogies = analogies[:args.max_analogies]
                analogy_ds = AnalogyDataset(analogies, cat_to_idx)
                analogy_loader = DataLoader(
                    analogy_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, collate_fn=collate_fn,
                )
                e3 = evaluate_e3_analogy(model, analogy_loader, cat_to_idx, args.device)
                for k, v in e3.items():
                    ckpt_results[f"{split_name}/{k}"] = v
                print(f"    E3 Direct cos={e3['e3_direct_cosine']:.4f} "
                      f"RetTop1={e3['e3_retrieval_top1']:.1%} "
                      f"RetTop3={e3['e3_retrieval_top3']:.1%}")
            else:
                print(f"    E3 Analogy: no data for {split_name}")

        elapsed = time.time() - t0
        print(f"\n  Completed in {elapsed:.1f}s")
        all_results[ckpt_name] = ckpt_results

    # ---- Summary Comparison ----
    print("\n" + "=" * 70)
    print("C O M P A R I S O N   S U M M A R Y")
    print("=" * 70)

    key_metrics = [
        ("health/eff_rank", "Eff Rank"),
        ("health/d90", "D90"),
        ("health/attr_OBJ_var", "OBJ Var%"),
        ("health/attr_COL_var", "COL Var%"),
        ("health/attr_SIZE_var", "SIZE Var%"),
        ("health/attr_MAT_var", "MAT Var%"),
    ]

    # Add per-split metrics
    for split in ["test_ood", "test_iid", "val", "train"]:
        key_metrics.append((f"{split}/e1_top1", f"{split} E1 Top1"))
        key_metrics.append((f"{split}/e2_mean_cosine", f"{split} E2 Cos"))
        key_metrics.append((f"{split}/e3_direct_cosine", f"{split} E3 Cos"))
        key_metrics.append((f"{split}/e3_retrieval_top1", f"{split} E3 RetTop1"))

    # Header
    header = f"{'Metric':<25}"
    for name in all_results:
        header += f" {name:>30}"
    print(header)
    print("-" * len(header))

    for metric_key, label in key_metrics:
        row = f"{label:<25}"
        for name in all_results:
            val = all_results[name].get(metric_key, float('nan'))
            if isinstance(val, float):
                row += f" {val:>30.4f}"
            else:
                row += f" {str(val):>30}"
        print(row)

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")

    writer.close()


if __name__ == "__main__":
    main()
