"""Wave 1 Completion: L_analogy gap + Full-Corpus Retrieval.

Task 1: Measure L_analogy train/val gap to verify anti-shortcut effectiveness.
Task 2: Full-corpus retrieval replacing batch-64 E3 evaluation.

Usage:
    python scripts/wave1_completion.py
"""

import json, sys, time
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
from sam.data.dataset import build_vocabs, SceneDataset, AnalogyDataset, collate_fn
from sam.losses.analogy import AnalogyLoss
from torch.utils.data import DataLoader


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def load_fix_model(checkpoint_path, device):
    """Load fixed model (512-dim symbol MLP) from checkpoint."""
    cfg = Config()
    cfg.model.vit_pretrained = False
    cat_to_idx, cat_sizes, idx_to_token = build_vocabs(cfg.data)
    visual = VisualEncoder(cfg.model, 256)
    symbol = SymbolEncoder(cat_sizes, 64, 512, 256)
    model = SAMPipeline(visual, symbol, 256).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    md = model.state_dict()
    pd = {k: v for k, v in ckpt["model_state_dict"].items()
          if k in md and md[k].shape == v.shape}
    md.update(pd)
    model.load_state_dict(md)
    model.eval()
    return model, cat_to_idx, cat_sizes, idx_to_token, cfg


# =============================================================================
# Task 1: L_analogy Train/Val Gap
# =============================================================================

def measure_analogy_gap(model, data_dir, cat_to_idx, device):
    print("=" * 60)
    print("Task 1: L_analogy Train/Val Gap")
    print("=" * 60)

    loss_fn = AnalogyLoss()

    results = {}
    for split_name, meta_file in [("train", "analogy_train_meta.json"),
                                    ("val", "analogy_val_meta.json")]:
        with open(Path(data_dir) / meta_file) as f:
            meta = json.load(f)
        dataset = AnalogyDataset(meta, cat_to_idx)
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            collate_fn=collate_fn)

        losses = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  Analogy {split_name}"):
                batch = to_device(batch, device)
                z_va = model.encode_visual(batch["img_a"])
                z_vb = model.encode_visual(batch["img_b"])
                z_sa = model.encode_symbol(batch["sym_a"])
                z_sb = model.encode_symbol(batch["sym_b"])
                l = loss_fn(z_va, z_vb, z_sa, z_sb).item()
                losses.append(l)

        avg_loss = np.mean(losses)
        results[split_name] = {
            "n_batches": len(losses),
            "n_samples": len(meta),
            "mean_loss": avg_loss,
        }
        print(f"  {split_name}: loss={avg_loss:.6f} (n={len(meta)})")

    gap = results["train"]["mean_loss"] / max(results["val"]["mean_loss"], 1e-8)
    print(f"\n  Train/Val ratio: {gap:.2f}x")
    print(f"  Threshold (<5x): {'PASS' if gap < 5 else 'FAIL'}")
    return results, gap


# =============================================================================
# Task 2: Full-Corpus Retrieval
# =============================================================================

def build_candidate_corpus(model, data_dir, cat_to_idx, device):
    """Precompute symbol embeddings for all unique candidates.

    Corpus includes:
      - All 360 unique single-object combos (5×6×3×4)
      - All 5000 train scene symbols
    """
    print("\nBuilding candidate corpus...")

    cfg = Config()
    colors = cfg.data.colors
    shapes = cfg.data.objects
    sizes = cfg.data.sizes
    materials = cfg.data.materials

    all_symbols = []

    # Single objects: all unique combos
    for shape in shapes:
        for color in colors:
            for size in sizes:
                for mat in materials:
                    all_symbols.append(
                        f"[OBJ:{shape}] [COL:{color}] [SIZE:{size}] [MAT:{mat}]"
                    )
    n_single = len(all_symbols)
    print(f"  Single-object combos: {n_single}")

    # Scenes: use all train scenes
    with open(Path(data_dir) / "scenes_train_meta.json") as f:
        scenes_train = json.load(f)
    scene_symbols = set()
    for scene in scenes_train:
        scene_symbols.add(scene["symbol_string"])
    all_symbols.extend(sorted(scene_symbols))
    n_scene = len(scene_symbols)
    print(f"  Unique scene symbols: {n_scene}")
    print(f"  Total corpus size: {len(all_symbols)}")

    # Precompute embeddings — process single-objects and scenes separately
    # (they have different token dimensions: 1 obj vs 2 obj)
    from sam.data.dataset import tokenize
    corpus_embeddings = []
    corpus_symbols = []

    def encode_symbol_batch(symbols, desc):
        """Encode a list of symbol strings in batches."""
        embs = []
        for i in range(0, len(symbols), 128):
            batch = symbols[i:i + 128]
            all_toks = {cat: [] for cat in cat_to_idx}
            for sym in batch:
                tok = tokenize(sym, cat_to_idx)
                for cat in all_toks:
                    t = tok[cat]
                    if t.dim() == 0:
                        t = t.unsqueeze(0)
                    all_toks[cat].append(t)
            batch_tokens = {cat: torch.stack(vals).to(device) for cat, vals in all_toks.items()}
            with torch.no_grad():
                z_s = model.encode_symbol(batch_tokens)
            embs.append(z_s.cpu())
        return torch.cat(embs, dim=0) if embs else torch.empty(0, 256)

    # Single objects
    single_syms = sorted(set(all_symbols[:n_single]))
    print(f"  Encoding {len(single_syms)} single-object symbols...")
    single_emb = encode_symbol_batch(single_syms, "single")
    corpus_embeddings.append(single_emb)
    corpus_symbols.extend(single_syms)

    # Scene symbols
    print(f"  Encoding {len(scene_symbols)} scene symbols...")
    scene_emb = encode_symbol_batch(sorted(scene_symbols), "scene")
    corpus_embeddings.append(scene_emb)
    corpus_symbols.extend(sorted(scene_symbols))

    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)  # (C, 256)
    corpus_embeddings = corpus_embeddings / corpus_embeddings.norm(dim=1, keepdim=True)
    print(f"  Corpus embeddings: {corpus_embeddings.shape}")

    return corpus_embeddings, corpus_symbols


def full_corpus_retrieval(model, data_dir, cat_to_idx, device):
    print("\n" + "=" * 60)
    print("Task 2: Full-Corpus Retrieval (E3 replacement)")
    print("=" * 60)

    # Build corpus
    corpus_emb, corpus_syms = build_candidate_corpus(
        model, data_dir, cat_to_idx, device)

    # Load OOD analogy queries
    with open(Path(data_dir) / "analogy_test_ood_meta.json") as f:
        ood_analogies = json.load(f)

    dataset = AnalogyDataset(ood_analogies, cat_to_idx)
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        collate_fn=collate_fn)

    top1, top3, top10, top100 = 0, 0, 0, 0
    total = 0
    cosine_margins = []

    corpus_gpu = corpus_emb.to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating analogies"):
            batch = to_device(batch, device)
            z_va = model.encode_visual(batch["img_a"])
            z_vb = model.encode_visual(batch["img_b"])
            z_sa = model.encode_symbol(batch["sym_a"])
            z_sb_true = model.encode_symbol(batch["sym_b"])

            z_sb_pred = z_vb - z_va + z_sa
            z_sb_pred = torch.nn.functional.normalize(z_sb_pred, dim=-1)

            B = z_sb_pred.shape[0]
            sim_matrix = torch.matmul(z_sb_pred, corpus_gpu.T)  # (B, C)

            for i in range(B):
                sims = sim_matrix[i]
                # Look up ground truth symbol in corpus
                true_sym = ood_analogies[total + i]["sym_b"]
                try:
                    true_idx = corpus_syms.index(true_sym)
                except ValueError:
                    z_true = z_sb_true[i:i+1]
                    cos_to_true = torch.matmul(z_true, corpus_gpu.T).squeeze(0)
                    true_idx = cos_to_true.argmax().item()

                true_score = sims[true_idx].item()
                higher = (sims > true_score).sum().item()
                rank = higher + 1

                if rank == 1: top1 += 1
                if rank <= 3: top3 += 1
                if rank <= 10: top10 += 1
                if rank <= 100: top100 += 1

                sorted_sims, sorted_idx = sims.sort(descending=True)
                if sorted_idx[0].item() == true_idx:
                    margin = sorted_sims[0].item() - sorted_sims[1].item()
                else:
                    margin = sorted_sims[0].item() - true_score
                cosine_margins.append(margin)

            total += B

    print(f"\n  Total queries: {total}")
    print(f"  Top-1:  {top1}/{total} ({top1/total:.1%})")
    print(f"  Top-3:  {top3}/{total} ({top3/total:.1%})")
    print(f"  Top-10: {top10}/{total} ({top10/total:.1%})")
    print(f"  Top-100: {top100}/{total} ({top100/total:.1%})")
    print(f"  Mean margin: {np.mean(cosine_margins):.4f}")
    print(f"\n  Thresholds:")
    print(f"    Top-10 > 50%: {'PASS' if top10/total > 0.5 else 'FAIL'}")

    # Compare with batch-64 baseline
    # Re-run batch-64 retrieval for comparison
    batch_top1, batch_top3, batch_total = 0, 0, 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=64, shuffle=False,
                                 collate_fn=collate_fn):
            batch = to_device(batch, device)
            z_va = model.encode_visual(batch["img_a"])
            z_vb = model.encode_visual(batch["img_b"])
            z_sa = model.encode_symbol(batch["sym_a"])
            z_sb_true = model.encode_symbol(batch["sym_b"])
            z_sb_pred = z_vb - z_va + z_sa
            z_sb_pred = torch.nn.functional.normalize(z_sb_pred, dim=-1)

            B = z_sb_pred.shape[0]
            cos = torch.matmul(z_sb_pred, z_sb_true.T)
            _, top1_idx = cos.max(dim=1)
            batch_top1 += (top1_idx == torch.arange(B, device=device)).sum().item()
            _, top3_idx = cos.topk(min(3, B), dim=1)
            for i in range(B):
                if i in top3_idx[i]:
                    batch_top3 += 1
            batch_total += B

    print(f"\n  Batch-64 comparison:")
    print(f"    Top-1: {batch_top1}/{batch_total} ({batch_top1/batch_total:.1%})")
    print(f"    Top-3: {batch_top3}/{batch_total} ({batch_top3/batch_total:.1%})")

    results = {
        "corpus_size": len(corpus_syms),
        "n_queries": total,
        "top1": top1 / total,
        "top3": top3 / total,
        "top10": top10 / total,
        "top100": top100 / total,
        "mean_margin": float(np.mean(cosine_margins)),
        "batch64_top1": batch_top1 / batch_total,
        "batch64_top3": batch_top3 / batch_total,
    }
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    device = "cuda"
    data_dir = "data/sam_dataset"
    checkpoint = "outputs/p2_wave1_fix/checkpoint_epoch030.pt"

    print(f"Loading model: {checkpoint}")
    model, cat_to_idx, cat_sizes, idx_to_token, cfg = load_fix_model(
        checkpoint, device)
    print(f"Model loaded.")

    # Task 1: Analogy gap
    gap_results, gap_ratio = measure_analogy_gap(model, data_dir, cat_to_idx, device)

    # Task 2: Full-corpus retrieval
    retrieval_results = full_corpus_retrieval(model, data_dir, cat_to_idx, device)

    # Save results
    output = {
        "task1_analogy_gap": {
            "train_loss": gap_results["train"]["mean_loss"],
            "val_loss": gap_results["val"]["mean_loss"],
            "val_train_ratio": float(gap_results["val"]["mean_loss"] / max(gap_results["train"]["mean_loss"], 1e-8)),
            "pass_5x": (gap_results["val"]["mean_loss"] / max(gap_results["train"]["mean_loss"], 1e-8)) < 5,
        },
        "task2_full_corpus_retrieval": retrieval_results,
    }

    output_path = Path("outputs/p2_wave1_fix/wave1_completion.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Wave 1 Completion Summary")
    print(f"{'='*60}")
    print(f"  Task 1 (L_analogy gap): {gap_ratio:.2f}x {'PASS' if gap_ratio < 5 else 'FAIL'}")
    print(f"  Task 2 (Full-corpus Top-10): {retrieval_results['top10']:.1%} "
          f"{'PASS' if retrieval_results['top10'] > 0.5 else 'FAIL'}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
