"""Phase 2 Wave 1: Structural Fix Training.

Fixes two core issues identified in Phase 1:
  1. Color blindness — L_color-contrast + balanced sampling
  2. Alignment shortcut — 3-phase anti-shortcut schedule + L_perturb

Resumes from P3 best checkpoint.

Usage:
    python scripts/p2_wave1_train.py --p3_dir outputs/p3 --output_dir outputs/p2_wave1
"""

import argparse, json, sys, time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline
from sam.data.dataset import (
    build_vocabs, SceneDataset, AnalogyDataset, SingleObjectDataset, collate_fn,
)
from sam.data.balanced_sampler import ColorBalancedSampler
from sam.losses.align import AlignmentLoss
from sam.losses.relational import RelationalConsistencyLoss
from sam.losses.analogy import AnalogyLoss
from sam.losses.color_contrastive import ColorContrastiveLoss
from sam.losses.perturb import PerturbationLoss


# ---------------------------------------------------------------------------
#  Anti-Shortcut Phase Scheduler
# ---------------------------------------------------------------------------

class AntiShortcutScheduler:
    """Three-phase training to break the alignment shortcut.

    A: Recovery (gentle L_align, no analogy)
    B: Analogy Push (strong L_analogy, weak L_align, L_perturb active)
    C: Joint Finetune (balanced, all losses)
    """

    def __init__(self, epochs_a=5, epochs_b=15, epochs_c=10):
        self.boundaries = {
            "recovery": (1, epochs_a),
            "analogy": (epochs_a + 1, epochs_a + epochs_b),
            "joint": (epochs_a + epochs_b + 1, epochs_a + epochs_b + epochs_c),
        }

    def get_phase(self, epoch):
        for phase, (lo, hi) in self.boundaries.items():
            if lo <= epoch <= hi:
                return phase
        return "joint"

    def get_loss_weights(self, phase):
        if phase == "recovery":
            return {"align": 0.3, "rel": 1.0, "analogy": 0.0,
                    "disentangle": 0.0, "color_contrast": 0.5, "perturb": 0.0}
        elif phase == "analogy":
            return {"align": 0.1, "rel": 0.5, "analogy": 2.0,
                    "disentangle": 0.1, "color_contrast": 0.3, "perturb": 0.3}
        else:  # joint
            return {"align": 0.5, "rel": 1.0, "analogy": 1.0,
                    "disentangle": 0.1, "color_contrast": 0.2, "perturb": 0.0}

    def total_epochs(self):
        return self.boundaries["joint"][1]


# ---------------------------------------------------------------------------
#  Training helpers
# ---------------------------------------------------------------------------

def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def get_shape_color_ids(batch, cat_to_idx):
    """Extract shape and color indices from token batch."""
    tokens = batch["tokens"]
    shape_ids = tokens["OBJ"][:, 0]  # (B,) first object's shape
    color_ids = tokens["COL"][:, 0]  # (B,) first object's color
    return shape_ids, color_ids


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p3_dir", type=str, default="outputs/p3")
    parser.add_argument("--data_dir", type=str, default="data/sam_dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/p2_wave1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs_a", type=int, default=5,
                        help="Recovery phase epochs")
    parser.add_argument("--epochs_b", type=int, default=15,
                        help="Analogy push phase epochs")
    parser.add_argument("--epochs_c", type=int, default=10,
                        help="Joint finetune phase epochs")
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.model.vit_pretrained = False

    scheduler = AntiShortcutScheduler(args.epochs_a, args.epochs_b, args.epochs_c)
    total_epochs = scheduler.total_epochs()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Phase 2 Wave 1: Structural Fixes")
    print(f"{'='*60}")
    print(f"Phases: recovery(1-{args.epochs_a}) → "
          f"analogy({args.epochs_a+1}-{args.epochs_a+args.epochs_b}) → "
          f"joint({args.epochs_a+args.epochs_b+1}-{total_epochs})")
    print(f"Device: {args.device}")

    # Vocab
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

    # Load data
    with open(data_dir / "single_objects_meta.json") as f:
        single_meta = json.load(f)
    with open(data_dir / "scenes_train_meta.json") as f:
        scenes_train = json.load(f)
    with open(data_dir / "scenes_val_meta.json") as f:
        scenes_val = json.load(f)
    with open(data_dir / "analogy_train_meta.json") as f:
        analogy_train = json.load(f)
    with open(data_dir / "analogy_val_meta.json") as f:
        analogy_val = json.load(f)

    batch_size = 12  # increased from 8 for better color coverage

    # Scene loader with color-balanced sampling
    scene_train_ds = SceneDataset(scenes_train, cat_to_idx)
    color_sampler = ColorBalancedSampler(scenes_train, batch_size=batch_size)

    scene_train_loader = DataLoader(
        scene_train_ds, batch_sampler=color_sampler,
        num_workers=0, collate_fn=collate_fn,
    )

    scene_val_loader = DataLoader(
        SceneDataset(scenes_val, cat_to_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # Analogy loader
    analogy_train_ds = AnalogyDataset(analogy_train, cat_to_idx)
    analogy_train_loader = DataLoader(
        analogy_train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )

    analogy_val_loader = DataLoader(
        AnalogyDataset(analogy_val, cat_to_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    n_train = len(scenes_train)
    print(f"Train scenes: {n_train}, Val scenes: {len(scenes_val)}")
    print(f"Train analogies: {len(analogy_train)}, Batch size: {batch_size}")

    # Model
    visual = VisualEncoder(cfg.model, cfg.model.manifold_dim)
    symbol = SymbolEncoder(cat_sizes, cfg.model.symbol_embed_dim,
                           cfg.model.symbol_hidden_dim,
                           cfg.model.manifold_dim)
    model = SAMPipeline(visual, symbol, cfg.model.manifold_dim).to(args.device)

    # Load P3 checkpoint
    p3_ckpt_path = Path(args.p3_dir) / "checkpoint_best.pt"
    ckpt = torch.load(str(p3_ckpt_path), map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded P3 checkpoint from {p3_ckpt_path}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} total, {n_trainable:,} trainable")

    # Loss modules
    loss_align = AlignmentLoss(temperature=0.07)
    loss_rel = RelationalConsistencyLoss()
    loss_analogy = AnalogyLoss()
    loss_color = ColorContrastiveLoss(temperature=0.1)
    loss_perturb = PerturbationLoss(perturb_prob=0.3, margin=0.5)

    # Optimizer (fresh — don't load optimizer state from P3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6,
    )

    rng = np.random.RandomState(42)
    best_val = float("inf")

    print(f"\n{'='*60}")
    print(f"Training {total_epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, total_epochs + 1):
        phase = scheduler.get_phase(epoch)
        weights = scheduler.get_loss_weights(phase)
        model.train()
        t0 = time.time()

        metrics = {"loss_align": 0, "loss_rel": 0, "loss_analogy": 0,
                    "loss_color": 0, "loss_perturb": 0}
        n_batches = 0

        # Create analogy iterator
        analogy_iter = iter(analogy_train_loader) if weights["analogy"] > 0 else None

        pbar = tqdm(scene_train_loader, desc=f"E{epoch:3d} [{phase:8s}]")
        for batch in pbar:
            batch = to_device(batch, args.device)
            total_loss = torch.tensor(0.0, device=args.device)

            # --- Visual + Symbol encodings ---
            z_v = model.encode_visual(batch["image"])
            z_s, per_cat = model.encode_symbol(batch["tokens"],
                                                return_per_category=True)

            # --- L_align ---
            if weights["align"] > 0:
                la = loss_align(z_v, z_s)
                total_loss = total_loss + weights["align"] * la
                metrics["loss_align"] += la.item()

            # --- L_rel (within-batch pairs) ---
            if weights["rel"] > 0 and z_v.shape[0] >= 2:
                mid = z_v.shape[0] // 2
                if mid > 0:
                    lr_ = loss_rel(z_v[:mid], z_v[mid:2*mid],
                                   z_s[:mid], z_s[mid:2*mid])
                    total_loss = total_loss + weights["rel"] * lr_
                    metrics["loss_rel"] += lr_.item()

            # --- L_color-contrast ---
            if weights["color_contrast"] > 0:
                shape_ids, color_ids = get_shape_color_ids(batch, cat_to_idx)
                lc = loss_color(z_v, shape_ids, color_ids)
                if lc.item() > 0:
                    total_loss = total_loss + weights["color_contrast"] * lc
                    metrics["loss_color"] += lc.item()

            # --- Analogy step (interleaved) ---
            if analogy_iter and weights["analogy"] > 0:
                try:
                    a_batch = next(analogy_iter)
                except StopIteration:
                    analogy_iter = iter(analogy_train_loader)
                    a_batch = next(analogy_iter)

                a_batch = to_device(a_batch, args.device)
                z_va = model.encode_visual(a_batch["img_a"])
                z_vb = model.encode_visual(a_batch["img_b"])
                z_sa, _ = model.encode_symbol(a_batch["sym_a"],
                                               return_per_category=True)
                z_sb, _ = model.encode_symbol(a_batch["sym_b"],
                                               return_per_category=True)

                # L_analogy
                la_ = loss_analogy(z_va, z_vb, z_sa, z_sb)
                total_loss = total_loss + weights["analogy"] * la_
                metrics["loss_analogy"] += la_.item()

                # L_perturb: penalize alignment shortcut
                if weights["perturb"] > 0:
                    p_mask, p_tokens = loss_perturb(
                        z_vb, z_va, z_sa, z_sb, a_batch["sym_a"],
                        cat_sizes, rng=rng,
                    )
                    # Re-encode perturbed symbols
                    z_sa_perturbed = model.encode_symbol(p_tokens)
                    # If model still predicts S_B well with perturbed S_A,
                    # that's the shortcut — penalize it
                    z_pred_perturbed = z_vb - z_va + z_sa_perturbed
                    z_pred_perturbed = torch.nn.functional.normalize(
                        z_pred_perturbed, dim=-1)
                    cos_perturbed = (z_pred_perturbed * z_sb).sum(dim=-1)
                    # Penalize when cosine exceeds margin
                    violation = torch.clamp(
                        cos_perturbed - loss_perturb.margin, min=0)
                    lp = violation.mean()
                    total_loss = total_loss + weights["perturb"] * lp
                    metrics["loss_perturb"] += lp.item()

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_batches += 1
            postfix = {k: f"{v/n_batches:.4f}" for k, v in metrics.items()
                       if v > 0}
            pbar.set_postfix(postfix)

        scheduler_lr.step()

        # Average metrics
        for k in metrics:
            metrics[k] /= max(n_batches, 1)

        # Validate
        val_metrics = {}
        if epoch % 5 == 0 or epoch == total_epochs:
            model.eval()
            val_losses = {"val_align": 0, "val_color": 0}
            n_val = 0
            with torch.no_grad():
                for batch in scene_val_loader:
                    batch = to_device(batch, args.device)
                    zv = model.encode_visual(batch["image"])
                    zs = model.encode_symbol(batch["tokens"])
                    val_losses["val_align"] += loss_align(zv, zs).item()
                    shape_ids, color_ids = get_shape_color_ids(batch, cat_to_idx)
                    lc = loss_color(zv, shape_ids, color_ids)
                    if lc.item() > 0:
                        val_losses["val_color"] += lc.item()
                    n_val += 1
            for k in val_losses:
                val_metrics[k] = val_losses[k] / max(n_val, 1)

        # Print
        elapsed = time.time() - t0
        is_best = val_metrics.get("val_align", float("inf")) < best_val
        if is_best:
            best_val = val_metrics["val_align"]
        status = " [BEST]" if is_best else ""

        train_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if v > 0)
        val_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
        print(f"Epoch {epoch:3d} [{phase:8s}] {elapsed:.1f}s{status}")
        print(f"  Train: {train_str}")
        if val_str:
            print(f"  Val:   {val_str}")

        # Checkpoint
        ckpt_path = output_dir / f"checkpoint_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch, "phase": phase,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {**metrics, **val_metrics},
        }, str(ckpt_path))

        if is_best:
            import shutil
            shutil.copy2(str(ckpt_path), str(output_dir / "checkpoint_best.pt"))

    print(f"\nWave 1 training complete!")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
