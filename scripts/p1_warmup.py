"""P1 Warmup Training — Single-object cross-modal alignment.

Trains only L_align on single-object data to verify that visual and symbol
encoders produce nearby embeddings for the same object on the shared manifold.

Usage:
    python scripts/p1_warmup.py --data_dir data/test_mini --output_dir outputs/p1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline
from sam.data.dataset import SingleObjectDataset, build_vocabs, collate_fn


def to_device(obj, device):
    """Recursively move tensors to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def evaluate_alignment(model, loader, device):
    """Compute alignment metrics on single-object data.

    Returns:
        mean_cosine: average cosine similarity between matched visual/symbol pairs
        retrieval_acc: given visual embedding, top-1 retrieve correct symbol
    """
    model.eval()
    all_z_v = []
    all_z_s = []
    all_objects = []

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            z_v = model.encode_visual(batch["image"])
            z_s = model.encode_symbol(batch["tokens"])
            all_z_v.append(z_v)
            all_z_s.append(z_s)

    z_v = torch.cat(all_z_v, dim=0)  # (N, D)
    z_s = torch.cat(all_z_s, dim=0)  # (N, D)

    # Mean pairwise cosine similarity (diagonal = matched pairs)
    cosine_matrix = torch.matmul(z_v, z_s.T)  # (N, N)
    mean_cosine = cosine_matrix.diag().mean().item()

    # Retrieval accuracy: for each visual, find closest symbol
    # The correct answer is the diagonal element
    _, top1_idx = cosine_matrix.max(dim=1)  # (N,)
    correct = (top1_idx == torch.arange(len(z_v), device=device)).sum().item()
    retrieval_acc = correct / len(z_v)

    return mean_cosine, retrieval_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/test_mini")
    parser.add_argument("--output_dir", type=str, default="outputs/p1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default higher for warmup)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.lr_projection = args.lr
    cfg.model.vit_pretrained = True

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*50}")
    print(f"P1 Warmup — Single-object Alignment")
    print(f"{'='*50}")
    print(f"Device: {cfg.device}")
    print(f"Epochs: {args.epochs}")
    print(f"LR:     {args.lr}")
    print(f"Data:   {data_dir}")
    print(f"Output: {output_dir}")

    # Vocab
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

    # Data
    with open(data_dir / "single_objects_meta.json") as f:
        single_meta = json.load(f)
    dataset = SingleObjectDataset(single_meta, cat_to_idx)

    # 80/20 train/eval split
    n_train = int(len(dataset) * 0.8)
    n_eval = len(dataset) - n_train
    train_ds, eval_ds = torch.utils.data.random_split(
        dataset, [n_train, n_eval],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)
    print(f"Train samples: {n_train}, Eval samples: {n_eval}")

    # Model
    visual = VisualEncoder(cfg.model, manifold_dim=cfg.model.manifold_dim)
    symbol = SymbolEncoder(cat_sizes, embed_dim=cfg.model.symbol_embed_dim,
                           hidden_dim=cfg.model.symbol_hidden_dim,
                           manifold_dim=cfg.model.manifold_dim)
    model = SAMPipeline(visual, symbol, manifold_dim=cfg.model.manifold_dim)
    model = model.to(cfg.device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_total:,} total, {n_trainable:,} trainable")

    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # Loss with InfoNCE to prevent mode collapse
    temperature = 0.07
    def alignment_loss(z_v, z_s):
        # z_v, z_s are already normalized to unit sphere
        # Direct cosine loss on positive pairs
        cosine = (z_v * z_s).sum(dim=-1)
        direct_loss = (1.0 - cosine).mean()

        # InfoNCE: push non-matching pairs apart
        B = z_v.shape[0]
        logits = torch.matmul(z_v, z_s.T) / temperature  # (B, B)
        labels = torch.arange(B, device=z_v.device)
        infonce = (F.cross_entropy(logits, labels) +
                   F.cross_entropy(logits.T, labels)) / 2

        return direct_loss + 0.5 * infonce

    # Initial evaluation
    init_cosine, init_retrieval = evaluate_alignment(model, eval_loader, cfg.device)
    print(f"\nInitial (before training):")
    print(f"  Mean cosine sim: {init_cosine:.4f}")
    print(f"  Retrieval acc:   {init_retrieval:.4f}")
    print()

    # Training loop
    best_eval_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = to_device(batch, cfg.device)

            z_v = model.encode_visual(batch["image"])
            z_s = model.encode_symbol(batch["tokens"])

            loss = alignment_loss(z_v, z_s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Evaluate
        mean_cosine, retrieval_acc = evaluate_alignment(
            model, eval_loader, cfg.device
        )

        is_best = avg_loss < best_eval_loss
        best_eval_loss = min(best_eval_loss, avg_loss)

        marker = " *" if is_best else ""
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, "
              f"cosine={mean_cosine:.4f}, retrieval={retrieval_acc:.4f}{marker}")

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "cosine_sim": mean_cosine,
            "retrieval_acc": retrieval_acc,
        })

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": history[-1],
        }
        torch.save(ckpt, str(output_dir / f"p1_epoch{epoch:02d}.pt"))
        if is_best:
            torch.save(ckpt, str(output_dir / "p1_best.pt"))

    # Final report
    print(f"\n{'='*50}")
    print(f"P1 Warmup Results")
    print(f"{'='*50}")
    print(f"Initial cosine sim:  {init_cosine:.4f}")
    print(f"Initial retrieval:   {init_retrieval:.4f}")
    print(f"Final cosine sim:    {history[-1]['cosine_sim']:.4f}")
    print(f"Final retrieval:     {history[-1]['retrieval_acc']:.4f}")
    print(f"Improvement cosine:  {history[-1]['cosine_sim'] - init_cosine:+.4f}")
    print(f"Improvement retrieval:{history[-1]['retrieval_acc'] - init_retrieval:+.4f}")
    print(f"\nP1 Threshold (retrieval > 90%): "
          f"{'PASS' if history[-1]['retrieval_acc'] >= 0.9 else 'NOT YET'}")

    # Save report
    report = {
        "init_cosine": init_cosine,
        "init_retrieval": init_retrieval,
        "final_cosine": history[-1]["cosine_sim"],
        "final_retrieval": history[-1]["retrieval_acc"],
        "best_loss": best_eval_loss,
        "history": history,
    }
    with open(output_dir / "p1_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if cfg.device == "cuda":
        vram = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak VRAM: {vram:.1f} MB")


if __name__ == "__main__":
    main()
