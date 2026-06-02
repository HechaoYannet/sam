"""P2 Relation Phase Training.

Extends P1 alignment with relational consistency loss (L_rel) on dual-object scenes.
Uses the full trainer with TensorBoard, checkpointing, and curriculum support.

Usage:
    # Start TensorBoard in a separate terminal first:
    tensorboard --logdir outputs/p2/tensorboard --port 6006

    # Then run training:
    python scripts/p2_train.py --data_dir data/sam_dataset --output_dir outputs/p2 --epochs 20

    # Resume from checkpoint:
    python scripts/p2_train.py --data_dir data/sam_dataset --output_dir outputs/p2 --resume

    # In a third terminal, run the AI monitor:
    python scripts/monitor.py --output_dir outputs/p2 --interval 60
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
    parser = argparse.ArgumentParser(description="P2 Relation Training")
    parser.add_argument("--data_dir", type=str, default="data/sam_dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/p2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.train.epochs = args.epochs
    cfg.train.phase_warmup_end = 5
    cfg.train.phase_relation_end = args.epochs  # relation throughout P2
    cfg.train.phase_analogy_end = args.epochs    # no analogy in P2
    cfg.train.phase_finetune_end = args.epochs   # no finetune in P2
    if args.no_amp:
        cfg.train.use_amp = False

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"P2 Relation Training")
    print(f"{'='*60}")
    print(f"Device:    {cfg.device}")
    print(f"Data:      {data_dir}")
    print(f"Output:    {output_dir}")
    print(f"Epochs:    {cfg.train.epochs}")
    print(f"Phases:    warmup(1-5) → relation(6-{args.epochs})")
    print(f"Batch:     {cfg.train.micro_batch_size}x{cfg.train.gradient_accumulation_steps}={cfg.train.micro_batch_size * cfg.train.gradient_accumulation_steps}")
    print(f"AMP:       {cfg.train.use_amp}")
    print(f"TensorBoard: tensorboard --logdir {output_dir / 'tensorboard'} --port 6006")
    print(f"{'='*60}")

    # Vocab
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)
    print(f"Vocab sizes: {cat_sizes}")

    # --- Load data ---
    with open(data_dir / "single_objects_meta.json") as f:
        single_meta = json.load(f)

    # Scene data (required for P2)
    scenes_meta = {}
    for split in ["train", "val", "test_iid", "test_ood"]:
        path = data_dir / f"scenes_{split}_meta.json"
        if path.exists():
            with open(path) as f:
                scenes_meta[split] = json.load(f)

    if "train" not in scenes_meta or len(scenes_meta["train"]) < 100:
        print(f"\nERROR: Train scenes too few ({len(scenes_meta.get('train', []))}).")
        print("Run `python scripts/generate_data.py` first to generate the dataset.")
        sys.exit(1)

    # Build loaders
    dl_kwargs = dict(num_workers=cfg.train.num_workers, collate_fn=collate_fn)

    single_dataset = SingleObjectDataset(single_meta, cat_to_idx)
    single_loader = DataLoader(single_dataset,
                               batch_size=cfg.train.micro_batch_size,
                               shuffle=True, **dl_kwargs)

    scene_loaders = {}
    for split in ["train", "val", "test_iid", "test_ood"]:
        if split in scenes_meta:
            dataset = SceneDataset(scenes_meta[split], cat_to_idx)
            scene_loaders[split] = DataLoader(
                dataset, batch_size=cfg.train.micro_batch_size,
                shuffle=(split == "train"), **dl_kwargs,
            )

    train_scenes = len(scenes_meta.get("train", []))
    val_scenes = len(scenes_meta.get("val", []))
    print(f"Train scenes:  {train_scenes}")
    print(f"Val scenes:    {val_scenes}")
    print(f"Single objs:   {len(single_meta)}")

    # --- Model ---
    visual = VisualEncoder(cfg.model, manifold_dim=cfg.model.manifold_dim)
    symbol = SymbolEncoder(cat_sizes, embed_dim=cfg.model.symbol_embed_dim,
                           hidden_dim=cfg.model.symbol_hidden_dim,
                           manifold_dim=cfg.model.manifold_dim)
    model = SAMPipeline(visual, symbol, manifold_dim=cfg.model.manifold_dim)
    model = model.to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params:       {n_params:,} total, {n_trainable:,} trainable")

    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- Trainer ---
    trainer = SAMTrainer(model, cfg, cat_sizes, output_dir=str(output_dir))

    # Resume checkpoint
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

    # --- Training loop ---
    print(f"\nStarting P2 training (epochs {start_epoch}-{args.epochs})")
    print(f"Monitor: python scripts/monitor.py --output_dir {output_dir} --once")
    print()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            trainer.current_epoch = epoch
            t0 = time.time()
            phase = trainer._get_phase(epoch)

            # Use single_loader during warmup, scene_loader during relation
            if phase == "warmup":
                primary_loader = single_loader
            else:
                primary_loader = scene_loaders["train"]

            train_metrics = trainer.train_epoch(
                scene_loader=primary_loader,
                analogy_loader=None,  # no analogy in P2
            )

            # Validate
            val_metrics = {}
            if epoch % 5 == 0 and scene_loaders.get("val"):
                val_metrics = trainer.validate(
                    scene_loader=scene_loaders["val"],
                    analogy_loader=None,
                )

            trainer.scheduler.step()
            elapsed = time.time() - t0

            total_val = val_metrics.get("val_align", float("inf"))
            signals = trainer.check_improvement(total_val)
            is_best = signals["improved"]
            status = " [BEST]" if is_best else ""

            train_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
            print(f"Epoch {epoch:3d} [{phase:8s}] {elapsed:.1f}s{status}")
            print(f"  Train: {train_str}")
            if val_metrics:
                val_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                print(f"  Val:   {val_str}")

            trainer.save_checkpoint(
                str(output_dir / f"checkpoint_epoch{epoch:03d}.pt"),
                metrics={**train_metrics, **val_metrics},
                is_best=is_best,
            )

    except KeyboardInterrupt:
        print(f"\nInterrupted at epoch {trainer.current_epoch}. Saving checkpoint...")
        trainer.save_checkpoint(
            str(output_dir / f"checkpoint_interrupt_epoch{trainer.current_epoch:03d}.pt"),
            metrics=train_metrics,
        )
    finally:
        trainer.close()

    vram_str = ""
    if cfg.device == "cuda":
        vram_str = f", peak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB"

    print(f"\nP2 training complete!{vram_str}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"TensorBoard: tensorboard --logdir {output_dir / 'tensorboard'} --port 6006")


if __name__ == "__main__":
    main()
