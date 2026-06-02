"""P4 Evaluation — Compositional Systematicity Proof.

Three experiments that test whether the model has learned a systematic
shared manifold, or is just memorizing patterns.

E1: Compositional Generalization — IID vs OOD retrieval gap
E2: Cross-Modal Isomorphism — RSA between visual and symbol spaces
E3: Zero-Shot Cross-Modal Analogy — vector arithmetic on the manifold

Usage:
    python scripts/p4_eval.py --checkpoint outputs/p3/checkpoint_best.pt --data_dir data/sam_dataset
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline
from sam.data.dataset import (
    build_vocabs, SceneDataset, AnalogyDataset, collate_fn,
)
from torch.utils.data import DataLoader


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def load_model(checkpoint_path, cfg, cat_sizes, device):
    """Load trained model from checkpoint."""
    visual = VisualEncoder(cfg.model, manifold_dim=cfg.model.manifold_dim)
    symbol = SymbolEncoder(cat_sizes, embed_dim=cfg.model.symbol_embed_dim,
                           hidden_dim=cfg.model.symbol_hidden_dim,
                           manifold_dim=cfg.model.manifold_dim)
    model = SAMPipeline(visual, symbol, manifold_dim=cfg.model.manifold_dim)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
#  E1: Compositional Generalization
# ---------------------------------------------------------------------------

def eval_compositional_generalization(model, scene_meta_iid, scene_meta_ood,
                                       cat_to_idx, device):
    """Measure alignment + retrieval on IID vs OOD scene splits.

    The key metric is the IID-OOD gap. If the model composes systematically,
    the gap should be small (< 15%). A large gap indicates memorization.
    """
    results = {}
    for split_name, meta in [("IID", scene_meta_iid), ("OOD", scene_meta_ood)]:
        dataset = SceneDataset(meta, cat_to_idx)
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            collate_fn=collate_fn)

        all_z_v = []
        all_z_s = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"E1 {split_name}"):
                batch = to_device(batch, device)
                all_z_v.append(model.encode_visual(batch["image"]))
                all_z_s.append(model.encode_symbol(batch["tokens"]))

        z_v = torch.cat(all_z_v, dim=0)  # (N, D)
        z_s = torch.cat(all_z_s, dim=0)  # (N, D)

        # Pairwise cosine
        cosine_matrix = torch.matmul(z_v, z_s.T)  # (N, N)
        mean_cosine = cosine_matrix.diag().mean().item()

        # Retrieval: for each visual, find closest symbol
        _, top1 = cosine_matrix.max(dim=1)
        correct = (top1 == torch.arange(len(z_v), device=device)).sum().item()
        retrieval_acc = correct / len(z_v)

        results[split_name] = {
            "n_samples": len(z_v),
            "mean_cosine": mean_cosine,
            "retrieval_acc": retrieval_acc,
        }

    # Gap analysis
    gap_cosine = results["IID"]["mean_cosine"] - results["OOD"]["mean_cosine"]
    gap_retrieval = results["IID"]["retrieval_acc"] - results["OOD"]["retrieval_acc"]

    print(f"\n{'='*50}")
    print(f"E1: Compositional Generalization")
    print(f"{'='*50}")
    print(f"  IID:  cosine={results['IID']['mean_cosine']:.4f}, retrieval={results['IID']['retrieval_acc']:.4f}")
    print(f"  OOD:  cosine={results['OOD']['mean_cosine']:.4f}, retrieval={results['OOD']['retrieval_acc']:.4f}")
    print(f"  Gap:  cosine={gap_cosine:.4f}, retrieval={gap_retrieval:.4f}")
    print(f"  Threshold (retrieval gap < 0.15): {'PASS' if gap_retrieval < 0.15 else 'FAIL'}")

    results["gap_cosine"] = gap_cosine
    results["gap_retrieval"] = gap_retrieval
    return results


# ---------------------------------------------------------------------------
#  E2: Cross-Modal Isomorphism (RSA)
# ---------------------------------------------------------------------------

def eval_isomorphism(model, scene_meta, cat_to_idx, device, n_samples=500):
    """RSA between visual and symbol representation spaces.

    For n_samples scenes, compute visual RDM and symbol RDM, then measure
    their Spearman correlation. High correlation = cross-modal isomorphism.
    """
    dataset = SceneDataset(scene_meta[:n_samples], cat_to_idx)
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        collate_fn=collate_fn)

    all_z_v = []
    all_z_s = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="E2 Encoding"):
            batch = to_device(batch, device)
            all_z_v.append(model.encode_visual(batch["image"]))
            all_z_s.append(model.encode_symbol(batch["tokens"]))

    z_v = torch.cat(all_z_v, dim=0)[:n_samples]  # (N, D)
    z_s = torch.cat(all_z_s, dim=0)[:n_samples]

    # Compute RDMs (Representational Dissimilarity Matrices)
    # RDM[i,j] = 1 - cosine_sim(zi, zj)
    def compute_rdm(z):
        z_norm = z / z.norm(dim=1, keepdim=True)
        cos = torch.matmul(z_norm, z_norm.T)
        return (1.0 - cos).cpu().numpy()

    rdm_v = compute_rdm(z_v)
    rdm_s = compute_rdm(z_s)

    # Spearman correlation between upper triangles (excluding diagonal)
    n = rdm_v.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    rdm_v_flat = rdm_v[triu_idx]
    rdm_s_flat = rdm_s[triu_idx]

    from scipy.stats import spearmanr
    rho, pval = spearmanr(rdm_v_flat, rdm_s_flat)

    # Also compute relation displacement alignment
    # For random pairs, check if visual displacement aligns with symbol displacement
    n_pairs = min(200, n // 2)
    cos_sims = []
    rng = np.random.RandomState(42)
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        dv = z_v[i] - z_v[j]
        ds = z_s[i] - z_s[j]
        cos = (dv * ds).sum() / (dv.norm() * ds.norm() + 1e-8)
        cos_sims.append(cos.item())

    mean_displacement_cos = np.mean(cos_sims)

    print(f"\n{'='*50}")
    print(f"E2: Cross-Modal Isomorphism (RSA)")
    print(f"{'='*50}")
    print(f"  RSA Spearman ρ: {rho:.4f} (p={pval:.2e})")
    print(f"  Displacement cosine: {mean_displacement_cos:.4f}")
    print(f"  Threshold (ρ > 0.3): {'PASS' if rho > 0.3 else 'FAIL'}")

    return {
        "rsa_rho": rho,
        "rsa_pval": pval,
        "mean_displacement_cos": mean_displacement_cos,
        "n_pairs": n_pairs,
    }


# ---------------------------------------------------------------------------
#  E3: Zero-Shot Cross-Modal Analogy
# ---------------------------------------------------------------------------

def eval_zero_shot_analogy(model, analogy_meta_iid, analogy_meta_ood,
                            cat_to_idx, idx_to_token, cat_sizes, device):
    """Test analogy completion via manifold vector arithmetic.

    Given (img_A, sym_A, img_B), predict sym_B via:
        z_sb_pred = E_v(img_B) - E_v(img_A) + E_s(sym_A)

    Then find the closest symbol embedding to z_sb_pred.
    """
    results = {}
    for split_name, meta in [("IID", analogy_meta_iid),
                               ("OOD", analogy_meta_ood)]:
        dataset = AnalogyDataset(meta, cat_to_idx)
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            collate_fn=collate_fn)

        top1_correct = 0
        top3_correct = 0
        total = 0
        cosines = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"E3 {split_name}"):
                batch = to_device(batch, device)
                z_va = model.encode_visual(batch["img_a"])
                z_vb = model.encode_visual(batch["img_b"])
                z_sa = model.encode_symbol(batch["sym_a"])
                z_sb_true = model.encode_symbol(batch["sym_b"])

                # Analogy: z_sb_pred = z_vb - z_va + z_sa
                z_sb_pred = z_vb - z_va + z_sa
                z_sb_pred = torch.nn.functional.normalize(z_sb_pred, dim=-1)

                # Find closest symbol by generating all possible symbol embeddings
                # For efficiency, use the current batch as candidates
                # More thorough: enumerate all symbol combinations
                B = z_sb_pred.shape[0]

                # Within-batch retrieval as approximation
                cosine_matrix = torch.matmul(z_sb_pred, z_sb_true.T)  # (B, B)

                # Top-1
                _, top1_idx = cosine_matrix.max(dim=1)
                correct = (top1_idx == torch.arange(B, device=device)).sum().item()
                top1_correct += correct

                # Top-3
                _, top3_idx = cosine_matrix.topk(min(3, B), dim=1)
                for i in range(B):
                    if i in top3_idx[i]:
                        top3_correct += 1

                # Cosine on matched pairs
                cos = cosine_matrix.diag().mean().item()
                cosines.append(cos)

                total += B

        top1_acc = top1_correct / total
        top3_acc = top3_correct / total
        mean_cos = np.mean(cosines)

        results[split_name] = {
            "top1_acc": top1_acc,
            "top3_acc": top3_acc,
            "mean_cosine": mean_cos,
            "n_samples": total,
        }

    print(f"\n{'='*50}")
    print(f"E3: Zero-Shot Cross-Modal Analogy")
    print(f"{'='*50}")
    for split in ["IID", "OOD"]:
        r = results[split]
        print(f"  {split}:")
        print(f"    Top-1: {r['top1_acc']:.4f}  Top-3: {r['top3_acc']:.4f}  Cosine: {r['mean_cosine']:.4f}")
    gap = results["IID"]["top1_acc"] - results["OOD"]["top1_acc"]
    print(f"  OOD-IID gap: {gap:.4f}")
    print(f"  Threshold (OOD Top-1 > 30%): {'PASS' if results['OOD']['top1_acc'] > 0.3 else 'FAIL'}")

    results["gap_top1"] = gap
    return results


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/p3/checkpoint_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/sam_dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="outputs/p4_report.json")
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.model.vit_pretrained = False  # loading from checkpoint

    data_dir = Path(args.data_dir)
    cat_to_idx, cat_sizes, idx_to_token = build_vocabs(cfg.data)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, cfg, cat_sizes, args.device)
    print(f"Model loaded. Device: {args.device}")

    # Load data
    with open(data_dir / "scenes_test_iid_meta.json") as f:
        scenes_iid = json.load(f)
    with open(data_dir / "scenes_test_ood_meta.json") as f:
        scenes_ood = json.load(f)
    with open(data_dir / "analogy_test_iid_meta.json") as f:
        analogy_iid = json.load(f)
    with open(data_dir / "analogy_test_ood_meta.json") as f:
        analogy_ood = json.load(f)

    print(f"Data: IID scenes={len(scenes_iid)}, OOD scenes={len(scenes_ood)}")
    print(f"      IID analogies={len(analogy_iid)}, OOD analogies={len(analogy_ood)}")

    # Run experiments
    report = {}

    # E1
    e1 = eval_compositional_generalization(
        model, scenes_iid, scenes_ood, cat_to_idx, args.device
    )
    report["E1_compositional_generalization"] = e1

    # E2
    e2 = eval_isomorphism(
        model, scenes_iid, cat_to_idx, args.device, n_samples=min(500, len(scenes_iid))
    )
    report["E2_isomorphism"] = e2

    # E3
    e3 = eval_zero_shot_analogy(
        model, analogy_iid, analogy_ood, cat_to_idx, idx_to_token, cat_sizes, args.device
    )
    report["E3_zero_shot_analogy"] = e3

    # Final verdict
    print(f"\n{'='*60}")
    print(f"P4 Final Verdict")
    print(f"{'='*60}")

    checks = []
    checks.append(("E1: IID-OOD retrieval gap < 0.15",
                   e1.get("gap_retrieval", 1.0) < 0.15))
    checks.append(("E2: RSA ρ > 0.3",
                   e2.get("rsa_rho", 0) > 0.3))
    checks.append(("E3: OOD Top-1 analogy > 0.30",
                   e3.get("OOD", {}).get("top1_acc", 0) > 0.3))

    for desc, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}: {desc}")

    all_pass = all(p for _, p in checks)
    print(f"\n  Overall: {'H1 SUPPORTED' if all_pass else 'H1 PARTIALLY SUPPORTED' if sum(p for _,p in checks) >= 2 else 'H1 NOT SUPPORTED'}")

    # Save
    report["verdict"] = {
        "checks": [{"description": d, "passed": p} for d, p in checks],
        "all_pass": all_pass,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    report = convert(report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
