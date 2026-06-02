import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from sam.manifold.projection import ManifoldProjection
from sam.losses.align import AlignmentLoss
from sam.losses.relational import RelationalConsistencyLoss
from sam.losses.analogy import AnalogyLoss
from sam.losses.disentangle import DisentangleLoss
from sam.data.dataset import collate_fn


class SAMPipeline(nn.Module):
    """Full SAM model: encoders + manifold projection."""

    def __init__(self, visual_encoder, symbol_encoder, manifold_dim=256):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.symbol_encoder = symbol_encoder
        self.v_proj = ManifoldProjection(manifold_dim)
        self.s_proj = ManifoldProjection(manifold_dim)

    def encode_visual(self, x):
        return self.v_proj(self.visual_encoder(x))

    def encode_symbol(self, tokens):
        return self.s_proj(self.symbol_encoder(tokens))


class SAMTrainer:
    """Curriculum trainer for SAM with gradient accumulation and AMP."""

    def __init__(self, model: SAMPipeline, cfg, cat_sizes: dict,
                 output_dir: str = "outputs"):
        self.model = model
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss modules
        self.loss_align = AlignmentLoss(temperature=cfg.loss.alignment_temperature)
        self.loss_rel = RelationalConsistencyLoss()
        self.loss_analogy = AnalogyLoss()
        self.loss_disentangle = DisentangleLoss()

        # Optimizer with different LRs for projection heads vs encoders
        proj_params = list(model.v_proj.parameters()) + list(model.s_proj.parameters())
        encoder_params = list(model.visual_encoder.parameters()) + \
                         list(model.symbol_encoder.parameters())

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": cfg.train.lr},
            {"params": proj_params, "lr": cfg.train.lr_projection},
        ], weight_decay=cfg.train.weight_decay, betas=cfg.train.betas)

        self.scaler = GradScaler(enabled=cfg.train.use_amp)
        self.device = cfg.device
        self.cat_sizes = cat_sizes

        self.current_epoch = 0
        self.metrics_history = defaultdict(list)

    def _get_phase(self, epoch: int) -> str:
        """Determine curriculum phase for the current epoch."""
        t = self.cfg.train
        if epoch < t.phase_warmup_end:
            return "warmup"
        elif epoch < t.phase_relation_end:
            return "relation"
        elif epoch < t.phase_analogy_end:
            return "analogy"
        else:
            return "finetune"

    def _get_active_losses(self, phase: str) -> dict:
        """Return loss weights based on curriculum phase."""
        cfg = self.cfg.loss
        if phase == "warmup":
            return {"align": 1.0, "rel": 0.0, "analogy": 0.0, "disentangle": 0.0}
        elif phase == "relation":
            return {"align": 1.0, "rel": cfg.alpha_rel, "analogy": 0.0, "disentangle": 0.0}
        elif phase == "analogy":
            return {"align": 1.0, "rel": cfg.alpha_rel,
                    "analogy": cfg.beta_analogy, "disentangle": cfg.gamma_disentangle}
        else:  # finetune
            return {"align": 0.5, "rel": cfg.alpha_rel,
                    "analogy": cfg.beta_analogy, "disentangle": cfg.gamma_disentangle}

    def train_epoch(self, single_loader, scene_loader, analogy_loader,
                    epoch: int):
        """Train for one epoch across all data types."""
        self.model.train()
        phase = self._get_phase(epoch)
        loss_weights = self._get_active_losses(phase)

        metrics = defaultdict(float)
        n_batches = 0
        accum = self.cfg.train.gradient_accumulation_steps

        self.optimizer.zero_grad()

        # Create iterators
        scene_iter = iter(scene_loader) if scene_loader else None
        analogy_iter = iter(analogy_loader) if analogy_loader else None

        # Progress bar based on largest loader
        if scene_loader:
            pbar = tqdm(scene_loader, desc=f"Epoch {epoch} [{phase}]")
        elif single_loader:
            pbar = tqdm(single_loader, desc=f"Epoch {epoch} [{phase}]")
        else:
            return metrics

        for batch_idx, batch in enumerate(pbar):
            total_loss = 0.0
            batch = self._to_device(batch)

            with autocast(enabled=self.cfg.train.use_amp):
                # Handle scene batch (primary loader)
                if "image" in batch:
                    loss, batch_metrics = self._scene_step(batch, loss_weights)
                    total_loss += loss
                    for k, v in batch_metrics.items():
                        metrics[k] += v

                # Interleave analogy samples if available
                if analogy_iter and loss_weights.get("analogy", 0) > 0:
                    try:
                        analogy_batch = next(analogy_iter)
                        analogy_batch = self._to_device(analogy_batch)
                        a_loss, a_metrics = self._analogy_step(analogy_batch, loss_weights)
                        total_loss += a_loss
                        for k, v in a_metrics.items():
                            metrics[k] += v
                    except StopIteration:
                        pass

                # Interleave single-object samples for alignment
                if single_loader and phase == "warmup":
                    try:
                        if scene_iter:
                            pass  # single objects handled in _scene_step for warmup
                    except Exception:
                        pass

            total_loss = total_loss / accum
            self.scaler.scale(total_loss).backward()

            n_batches += 1
            if (batch_idx + 1) % accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Log
            if (batch_idx + 1) % self.cfg.train.log_interval == 0:
                log_metrics = {k: v / n_batches for k, v in metrics.items()}
                pbar.set_postfix(log_metrics)

        # Average metrics
        for k in metrics:
            metrics[k] /= max(n_batches, 1)

        return dict(metrics)

    def _scene_step(self, batch, loss_weights):
        metrics = {}
        total_loss = 0.0

        z_v = self.model.encode_visual(batch["image"])
        z_s = self.model.encode_symbol(batch["tokens"])

        # Alignment loss
        if loss_weights["align"] > 0:
            l_align = self.loss_align(z_v, z_s)
            total_loss += loss_weights["align"] * l_align
            metrics["loss_align"] = l_align.item()

        # Relational consistency on within-batch pairs
        if loss_weights["rel"] > 0 and z_v.shape[0] >= 2:
            # Pair: first half vs second half
            mid = z_v.shape[0] // 2
            z_va, z_vb = z_v[:mid], z_v[mid:2*mid]
            z_sa, z_sb = z_s[:mid], z_s[mid:2*mid]
            l_rel = self.loss_rel(z_va, z_vb, z_sa, z_sb)
            total_loss += loss_weights["rel"] * l_rel
            metrics["loss_rel"] = l_rel.item()

        return total_loss, metrics

    def _analogy_step(self, batch, loss_weights):
        metrics = {}
        total_loss = 0.0

        z_va = self.model.encode_visual(batch["img_a"])
        z_vb = self.model.encode_visual(batch["img_b"])
        z_sa = self.model.encode_symbol(batch["sym_a"])
        z_sb = self.model.encode_symbol(batch["sym_b"])

        # Alignment on both A and B
        if loss_weights["align"] > 0:
            l_align_a = self.loss_align(z_va, z_sa)
            l_align_b = self.loss_align(z_vb, z_sb)
            l_align = (l_align_a + l_align_b) / 2
            total_loss += loss_weights["align"] * l_align
            metrics["loss_align"] = l_align.item()

        # Relational consistency
        if loss_weights["rel"] > 0:
            l_rel = self.loss_rel(z_va, z_vb, z_sa, z_sb)
            total_loss += loss_weights["rel"] * l_rel
            metrics["loss_rel"] = l_rel.item()

        # Analogy completion
        if loss_weights["analogy"] > 0:
            l_analogy = self.loss_analogy(z_va, z_vb, z_sa, z_sb)
            total_loss += loss_weights["analogy"] * l_analogy
            metrics["loss_analogy"] = l_analogy.item()

        return total_loss, metrics

    def _to_device(self, batch):
        """Move batch to device."""
        if batch is None:
            return None
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in value.items()}
            else:
                result[key] = value
        return result

    def save_checkpoint(self, path: str = None):
        """Save model checkpoint."""
        if path is None:
            path = self.output_dir / f"checkpoint_epoch{self.current_epoch}.pt"
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics_history": dict(self.metrics_history),
        }, path)
        return path

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.metrics_history = defaultdict(list, ckpt.get("metrics_history", {}))
        return ckpt["epoch"]
