"""SAM Training Script.

Usage:
    # New training
    python scripts/train.py --data_dir data/sam_dataset --output_dir outputs/run1

    # Resume from checkpoint
    python scripts/train.py --data_dir data/sam_dataset --output_dir outputs/run1 --resume

    # Resume from specific checkpoint
    python scripts/train.py --data_dir data/sam_dataset --output_dir outputs/run1 --resume checkpoint_epoch030.pt

    # Monitor with TensorBoard
    tensorboard --logdir outputs/run1/tensorboard
"""

import argparse
import json
import sys
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


def load_metadata(data_dir: Path):
    """Load all dataset metadata files."""
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

    return single_meta, scenes_meta, analogy_meta


def build_loaders(cfg, cat_to_idx, single_meta, scenes_meta, analogy_meta):
    """Build all DataLoaders."""
    train_kwargs = dict(
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        prefetch_factor=cfg.train.prefetch_factor,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    eval_kwargs = dict(
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        collate_fn=collate_fn,
    )

    # Single objects (warmup)
    single_dataset = SingleObjectDataset(single_meta, cat_to_idx)
    single_loader = DataLoader(
        single_dataset, batch_size=cfg.train.micro_batch_size,
        shuffle=True, **train_kwargs,
    )

    # Scene loaders (only train/val needed)
    scene_loaders = {}
    for split in ["train", "val"]:
        if split in scenes_meta:
            dataset = SceneDataset(scenes_meta[split], cat_to_idx)
            scene_loaders[split] = DataLoader(
                dataset, batch_size=cfg.train.micro_batch_size,
                shuffle=(split == "train"),
                **train_kwargs if split == "train" else eval_kwargs,
            )

    # Analogy loaders (only train/val needed)
    analogy_loaders = {}
    for split in ["train", "val"]:
        if split in analogy_meta and len(analogy_meta[split]) > 0:
            dataset = AnalogyDataset(analogy_meta[split], cat_to_idx)
            analogy_loaders[split] = DataLoader(
                dataset, batch_size=cfg.train.micro_batch_size,
                shuffle=(split == "train"),
                **train_kwargs if split == "train" else eval_kwargs,
            )

    return single_loader, scene_loaders, analogy_loaders


def main():
    parser = argparse.ArgumentParser(description="Train SAM model")
    parser.add_argument("--data_dir", type=str, default="data/sam_dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/run1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from checkpoint. 'auto' finds latest, or specify path.")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.train.epochs = args.epochs
    if args.no_amp:
        cfg.train.use_amp = False

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"SAM Training")
    print(f"{'='*60}")
    print(f"Device:    {cfg.device}")
    print(f"Data:      {data_dir}")
    print(f"Output:    {output_dir}")
    print(f"Epochs:    {cfg.train.epochs}")
    print(f"Batch:     {cfg.train.micro_batch_size} × {cfg.train.gradient_accumulation_steps} = {cfg.train.micro_batch_size * cfg.train.gradient_accumulation_steps}")
    print(f"AMP:       {cfg.train.use_amp}")
    print(f"{'='*60}")

    # Vocab
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)
    print(f"Vocab sizes: {cat_sizes}")

    # Data
    single_meta, scenes_meta, analogy_meta = load_metadata(data_dir)
    single_loader, scene_loaders, analogy_loaders = build_loaders(
        cfg, cat_to_idx, single_meta, scenes_meta, analogy_meta
    )

    print(f"Train scenes:     {len(scenes_meta.get('train', []))}")
    print(f"Val scenes:       {len(scenes_meta.get('val', []))}")
    print(f"Train analogies:  {len(analogy_meta.get('train', []))}")
    print(f"Val analogies:    {len(analogy_meta.get('val', []))}")

    # Model
    visual = VisualEncoder(cfg.model, manifold_dim=cfg.model.manifold_dim)
    symbol = SymbolEncoder(
        cat_sizes=cat_sizes,
        embed_dim=cfg.model.symbol_embed_dim,
        hidden_dim=cfg.model.symbol_hidden_dim,
        manifold_dim=cfg.model.manifold_dim,
    )
    model = SAMPipeline(visual, symbol, manifold_dim=cfg.model.manifold_dim)
    model = model.to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters:   {n_params:,} total, {n_trainable:,} trainable")

    # VRAM check
    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"VRAM:         {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB (initial)")
        print(f"TensorBoard:  tensorboard --logdir {output_dir / 'tensorboard'}")

    # Trainer
    trainer = SAMTrainer(model, cfg, cat_sizes, output_dir=str(output_dir))

    # Resume
    start_epoch = 1
    if args.resume is not None:
        if args.resume == "auto":
            resume_path = trainer.find_latest_checkpoint()
        else:
            resume_path = args.resume
            if not Path(resume_path).is_absolute():
                resume_path = output_dir / resume_path

        if resume_path and Path(str(resume_path)).exists():
            start_epoch = trainer.load_checkpoint(str(resume_path)) + 1
            print(f"Resumed from {resume_path}, restarting at epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {resume_path}, starting fresh")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch}")
    print(f"{'='*60}\n")

    try:
        for epoch in range(start_epoch, cfg.train.epochs + 1):
            trainer.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = trainer.train_epoch(
                scene_loader=scene_loaders.get("train"),
                analogy_loader=analogy_loaders.get("train"),
            )

            # Validate (every 5 epochs to save time, or every epoch if small dataset)
            val_metrics = {}
            do_val = (epoch % 5 == 0 or epoch == cfg.train.epochs or
                       len(scenes_meta.get("val", [])) < 500)
            if do_val and scene_loaders.get("val"):
                val_metrics = trainer.validate(
                    scene_loader=scene_loaders["val"],
                    analogy_loader=analogy_loaders.get("val"),
                )

            # Scheduler step
            trainer.scheduler.step()

            # Log epoch-level metrics
            epoch_time = time.time() - epoch_start
            trainer.writer.add_scalar("epoch/time_seconds", epoch_time, epoch)
            trainer.writer.add_scalar("epoch/lr", trainer.optimizer.param_groups[0]["lr"], epoch)
            for k, v in train_metrics.items():
                trainer.writer.add_scalar(f"epoch/{k}", v, epoch)

            # Monitoring & best model tracking
            total_val_loss = val_metrics.get("val_align", float("inf"))
            signals = trainer.check_improvement(total_val_loss)
            is_best = signals["improved"]

            # Print summary
            phase = trainer._get_phase(epoch)
            status = ""
            if is_best:
                status += " [BEST]"
            if signals.get("plateau_warning"):
                status += " [PLATEAU]"
            train_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
            val_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
            print(f"Epoch {epoch:3d} [{phase:8s}] {epoch_time:.1f}s{status}")
            print(f"  Train: {train_str}")
            if val_str:
                print(f"  Val:   {val_str}")
            if signals.get("suggestion"):
                print(f"  [MONITOR] {signals['suggestion']}")

            # Save checkpoint
            trainer.save_checkpoint(
                str(output_dir / f"checkpoint_epoch{epoch:03d}.pt"),
                metrics={**train_metrics, **val_metrics},
                is_best=is_best,
            )

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at epoch {trainer.current_epoch}.")
        print("Saving emergency checkpoint...")
        trainer.save_checkpoint(
            str(output_dir / f"checkpoint_interrupt_epoch{trainer.current_epoch:03d}.pt"),
            metrics=train_metrics,
        )
        print("Done. Resume later with --resume")

    finally:
        trainer.close()

    print(f"\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved in: {output_dir}")
    print(f"TensorBoard: tensorboard --logdir {output_dir / 'tensorboard'}")


if __name__ == "__main__":
    import time
    main()
