"""P3 Full 4-Loss Training.

Resumes from P2 checkpoint and trains with all four losses:
  - L_align: cross-modal alignment (cosine + InfoNCE)
  - L_rel: relational consistency on displacement vectors
  - L_analogy: analogy completion via manifold vector arithmetic
  - L_disentangle: cross-category orthogonality regularization

Usage:
    # Start TensorBoard first:
    tensorboard --logdir outputs/p3/tensorboard --port 6006

    # Train (resumes from P2 best checkpoint):
    python scripts/p3_train.py --p2_dir outputs/p2 --output_dir outputs/p3 --epochs 20

    # Or start fresh with full 4-loss curriculum:
    python scripts/p3_train.py --data_dir data/sam_dataset --output_dir outputs/p3 --epochs 40

    # Monitor:
    python scripts/monitor.py --output_dir outputs/p3 --once
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline, SAMTrainer
from sam.data.dataset import (
    SingleObjectDataset, SceneDataset, AnalogyDataset,
    build_vocabs, collate_fn,
)


def main():
    parser = argparse.ArgumentParser(description="P3 Full 4-Loss Training")
    parser.add_argument("--data_dir", type=str, default="data/sam_dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/p3")
    parser.add_argument("--p2_dir", type=str, default=None,
                        help="P2 output dir to resume from (uses checkpoint_best.pt)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs for P3 phase (default 20 = epochs 21-40)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from P3 checkpoint")
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.train.epochs = args.epochs
    # Set curriculum: relation→analogy phase
    cfg.train.phase_warmup_end = 0      # skip warmup when resuming
    cfg.train.phase_relation_end = 5    # brief relation tuning
    cfg.train.phase_analogy_end = max(5, args.epochs - 2)
    cfg.train.phase_finetune_end = args.epochs

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"P3 Full 4-Loss Training")
    print(f"{'='*60}")
    print(f"Device:    {cfg.device}")
    print(f"Data:      {data_dir}")
    print(f"Output:    {output_dir}")
    print(f"Epochs:    {args.epochs}")
    print(f"Losses:    L_align + L_rel + L_analogy + L_disentangle")
    print(f"Phases:    relation(1-{cfg.train.phase_relation_end}) → "
          f"analogy({cfg.train.phase_relation_end+1}-{cfg.train.phase_analogy_end}) → "
          f"finetune({cfg.train.phase_analogy_end+1}-{args.epochs})")
    print(f"TensorBoard: tensorboard --logdir {output_dir / 'tensorboard'} --port 6006")
    print(f"{'='*60}")

    # Vocab
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

    # Load data
    with open(data_dir / "single_objects_meta.json") as f:
        single_meta = json.load(f)

    scenes_meta = {}
    for split in ["train", "val", "test_iid", "test_ood"]:
        path = data_dir / f"scenes_{split}_meta.json"
        if path.exists():
            with open(path) as f:
                scenes_meta[split] = json.load(f)

    analogy_meta = {}
    for split in ["train", "val", "test_iid", "test_ood"]:
        path = data_dir / f"analogy_{split}_meta.json"
        if path.exists():
            with open(path) as f:
                analogy_meta[split] = json.load(f)

    # Build loaders
    dl_kwargs = dict(num_workers=cfg.train.num_workers, collate_fn=collate_fn)

    scene_loaders = {}
    for split in ["train", "val"]:
        if split in scenes_meta:
            dataset = SceneDataset(scenes_meta[split], cat_to_idx)
            scene_loaders[split] = DataLoader(
                dataset, batch_size=cfg.train.micro_batch_size,
                shuffle=(split == "train"), **dl_kwargs,
            )

    analogy_loaders = {}
    for split in ["train", "val"]:
        if split in analogy_meta and len(analogy_meta[split]) > 0:
            dataset = AnalogyDataset(analogy_meta[split], cat_to_idx)
            analogy_loaders[split] = DataLoader(
                dataset, batch_size=cfg.train.micro_batch_size,
                shuffle=(split == "train"), **dl_kwargs,
            )

    n_train_scenes = len(scenes_meta.get("train", []))
    n_train_analogies = len(analogy_meta.get("train", []))
    print(f"Train scenes:    {n_train_scenes}")
    print(f"Train analogies: {n_train_analogies}")

    if n_train_analogies < 100:
        print("\nERROR: Not enough analogy data. Run generate_data.py first.")
        sys.exit(1)

    # Model
    visual = VisualEncoder(cfg.model, manifold_dim=cfg.model.manifold_dim)
    symbol = SymbolEncoder(cat_sizes, embed_dim=cfg.model.symbol_embed_dim,
                           hidden_dim=cfg.model.symbol_hidden_dim,
                           manifold_dim=cfg.model.manifold_dim)
    model = SAMPipeline(visual, symbol, manifold_dim=cfg.model.manifold_dim)
    model = model.to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params:          {n_params:,} total")

    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Trainer
    trainer = SAMTrainer(model, cfg, cat_sizes, output_dir=str(output_dir))

    # Resume from P2 checkpoint
    start_epoch = 1
    resume_source = args.p2_dir or args.resume
    if args.p2_dir:
        p2_best = Path(args.p2_dir) / "checkpoint_best.pt"
        if p2_best.exists():
            start_epoch = trainer.load_checkpoint(str(p2_best)) + 1
            print(f"Loaded P2 checkpoint from {p2_best}")
            print(f"Resuming from epoch {start_epoch}")

    # Also check for explicit resume flag
    if args.resume and start_epoch == 1:
        if args.resume == "auto":
            resume_path = trainer.find_latest_checkpoint()
        else:
            resume_path = args.resume
            if not Path(resume_path).is_absolute():
                resume_path = output_dir / resume_path
        if resume_path and Path(str(resume_path)).exists():
            start_epoch = trainer.load_checkpoint(str(resume_path)) + 1

    # Adjust epoch range
    cfg.train.epochs = start_epoch + args.epochs - 1
    trainer.cfg.train.epochs = cfg.train.epochs

    # Training loop
    print(f"\nStarting P3 training (epochs {start_epoch}-{cfg.train.epochs})")
    print(f"Active losses: L_align + L_rel + L_analogy + L_disentangle")
    print()

    try:
        for epoch in range(start_epoch, cfg.train.epochs + 1):
            trainer.current_epoch = epoch
            t0 = time.time()
            phase = trainer._get_phase(epoch)

            train_metrics = trainer.train_epoch(
                scene_loader=scene_loaders["train"],
                analogy_loader=analogy_loaders["train"],
            )

            # Validate every 5 epochs
            val_metrics = {}
            do_val = (epoch % 5 == 0 or epoch == cfg.train.epochs)
            if do_val and scene_loaders.get("val"):
                val_metrics = trainer.validate(
                    scene_loader=scene_loaders["val"],
                    analogy_loader=analogy_loaders.get("val"),
                )

            trainer.scheduler.step()
            elapsed = time.time() - t0

            total_val = val_metrics.get("val_align", float("inf"))
            signals = trainer.check_improvement(total_val)
            is_best = signals["improved"]
            status = " [BEST]" if is_best else ""
            if signals.get("plateau_warning"):
                status += " [PLATEAU]"

            train_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
            print(f"Epoch {epoch:3d} [{phase:8s}] {elapsed:.1f}s{status}")
            print(f"  Train: {train_str}")
            if val_metrics:
                val_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                print(f"  Val:   {val_str}")
            if signals.get("suggestion"):
                print(f"  [MONITOR] {signals['suggestion']}")

            trainer.save_checkpoint(
                str(output_dir / f"checkpoint_epoch{epoch:03d}.pt"),
                metrics={**train_metrics, **val_metrics},
                is_best=is_best,
            )

    except KeyboardInterrupt:
        print(f"\nInterrupted at epoch {trainer.current_epoch}. Saving...")
        trainer.save_checkpoint(
            str(output_dir / f"checkpoint_interrupt_epoch{trainer.current_epoch:03d}.pt"),
            metrics=train_metrics,
        )
    finally:
        trainer.close()

    vram_str = ""
    if cfg.device == "cuda":
        vram_str = f", peak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB"

    print(f"\nP3 training complete!{vram_str}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"TensorBoard: tensorboard --logdir {output_dir / 'tensorboard'} --port 6006")


if __name__ == "__main__":
    main()
