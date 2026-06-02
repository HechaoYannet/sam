"""SAM Training Script.

Usage:
    python scripts/train.py --data_dir data/sam_dataset --output_dir outputs/run1
"""

import argparse
import json
import os
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


def main():
    parser = argparse.ArgumentParser(description="Train SAM model")
    parser.add_argument("--data_dir", type=str, default="data/sam_dataset",
                        help="Path to generated dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/run1",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.train.epochs = args.epochs

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {cfg.device}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Build vocabularies
    cat_to_idx, cat_sizes, idx_to_token = build_vocabs(cfg.data)
    print(f"Vocab sizes: {cat_sizes}")

    # Load metadata
    print("Loading metadata...")
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

    # Create datasets
    single_dataset = SingleObjectDataset(single_meta, cat_to_idx)
    single_loader = DataLoader(single_dataset, batch_size=cfg.train.micro_batch_size,
                               shuffle=True, num_workers=cfg.train.num_workers,
                               collate_fn=collate_fn)

    scene_loaders = {}
    for split in ["train", "val", "test_iid", "test_ood"]:
        if split in scenes_meta:
            dataset = SceneDataset(scenes_meta[split], cat_to_idx)
            shuffle = (split == "train")
            scene_loaders[split] = DataLoader(
                dataset, batch_size=cfg.train.micro_batch_size,
                shuffle=shuffle, num_workers=cfg.train.num_workers,
                collate_fn=collate_fn,
            )

    analogy_loaders = {}
    for split in ["train", "val", "test_iid", "test_ood"]:
        if split in analogy_meta and len(analogy_meta[split]) > 0:
            dataset = AnalogyDataset(analogy_meta[split], cat_to_idx)
            shuffle = (split == "train")
            analogy_loaders[split] = DataLoader(
                dataset, batch_size=cfg.train.micro_batch_size,
                shuffle=shuffle, num_workers=cfg.train.num_workers,
                collate_fn=collate_fn,
            )

    print(f"Single objects: {len(single_dataset)}")
    print(f"Train scenes: {len(scenes_meta.get('train', []))}")
    print(f"Train analogies: {len(analogy_meta.get('train', []))}")

    # Initialize model
    visual_encoder = VisualEncoder(cfg.model, manifold_dim=cfg.model.manifold_dim)
    symbol_encoder = SymbolEncoder(
        cat_sizes=cat_sizes,
        embed_dim=cfg.model.symbol_embed_dim,
        hidden_dim=cfg.model.symbol_hidden_dim,
        manifold_dim=cfg.model.manifold_dim,
    )
    model = SAMPipeline(
        visual_encoder=visual_encoder,
        symbol_encoder=symbol_encoder,
        manifold_dim=cfg.model.manifold_dim,
    )
    model = model.to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    # VRAM check: single forward pass
    print("\nVRAM check: running single forward pass...")
    model.eval()
    with torch.no_grad():
        # Get a sample batch from train loader
        sample_batch = next(iter(scene_loaders["train"]))
        sample_batch = {
            k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
            for k, v in sample_batch.items()
        }
        z_v = model.encode_visual(sample_batch["image"])
        z_s = model.encode_symbol(sample_batch["tokens"])
        print(f"  Visual output shape: {z_v.shape}")
        print(f"  Symbol output shape: {z_s.shape}")

        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Peak VRAM: {vram_mb:.1f} MB")
            if vram_mb < 4096:
                print(f"  [PASS] VRAM < 4GB ✓")
            else:
                print(f"  [WARN] VRAM > 4GB threshold")

    # Train
    trainer = SAMTrainer(
        model=model,
        cfg=cfg,
        cat_sizes=cat_sizes,
        output_dir=str(output_dir),
    )

    print(f"\nStarting training for {cfg.train.epochs} epochs...")
    for epoch in range(1, cfg.train.epochs + 1):
        trainer.current_epoch = epoch
        metrics = trainer.train_epoch(
            single_loader=single_loader,
            scene_loader=scene_loaders.get("train"),
            analogy_loader=analogy_loaders.get("train"),
            epoch=epoch,
        )

        if metrics:
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"Epoch {epoch}: {metric_str}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = trainer.save_checkpoint(
                str(output_dir / f"checkpoint_epoch{epoch}.pt")
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = trainer.save_checkpoint(str(output_dir / "checkpoint_final.pt"))
    print(f"\nTraining complete! Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
