import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sam.manifold.projection import ManifoldProjection
from sam.losses.align import AlignmentLoss
from sam.losses.relational import RelationalConsistencyLoss
from sam.losses.analogy import AnalogyLoss
from sam.losses.disentangle import DisentangleLoss


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

    def encode_symbol(self, tokens, return_per_category=False):
        if return_per_category:
            z, per_cat = self.symbol_encoder(tokens, return_per_category=True)
            return self.s_proj(z), per_cat
        return self.s_proj(self.symbol_encoder(tokens))


class SAMTrainer:
    """Curriculum trainer with TensorBoard, checkpointing, and monitoring."""

    def __init__(self, model: SAMPipeline, cfg, cat_sizes: dict,
                 output_dir: str = "outputs"):
        self.model = model
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        log_dir = self.output_dir / "tensorboard"
        log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Loss modules
        self.loss_align = AlignmentLoss(temperature=cfg.loss.alignment_temperature)
        self.loss_rel = RelationalConsistencyLoss()
        self.loss_analogy = AnalogyLoss()
        self.loss_disentangle = DisentangleLoss()

        # Optimizer
        proj_params = list(model.v_proj.parameters()) + list(model.s_proj.parameters())
        encoder_params = list(model.visual_encoder.parameters()) + \
                         list(model.symbol_encoder.parameters())

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": cfg.train.lr},
            {"params": proj_params, "lr": cfg.train.lr_projection},
        ], weight_decay=cfg.train.weight_decay, betas=cfg.train.betas)

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.train.epochs,
            eta_min=cfg.train.lr * 0.01,
        )

        self.scaler = GradScaler(enabled=cfg.train.use_amp)
        self.device = cfg.device
        self.cat_sizes = cat_sizes

        # State tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metrics_history = defaultdict(list)
        self.patience_counter = 0
        self.patience_limit = 15  # epochs without improvement before warning

        # Structured log
        self.log_path = self.output_dir / "training_log.jsonl"

        self._save_run_config()

    def _save_run_config(self):
        """Save run configuration for reproducibility."""
        config = {
            "manifold_dim": self.cfg.model.manifold_dim,
            "vit_model": self.cfg.model.vit_model,
            "lr": self.cfg.train.lr,
            "lr_projection": self.cfg.train.lr_projection,
            "weight_decay": self.cfg.train.weight_decay,
            "batch_size": self.cfg.train.micro_batch_size,
            "grad_accum": self.cfg.train.gradient_accumulation_steps,
            "epochs": self.cfg.train.epochs,
            "alpha_rel": self.cfg.loss.alpha_rel,
            "beta_analogy": self.cfg.loss.beta_analogy,
            "gamma_disentangle": self.cfg.loss.gamma_disentangle,
            "started_at": datetime.now().isoformat(),
        }
        with open(self.output_dir / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)

    def _get_phase(self, epoch: int) -> str:
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
        cfg = self.cfg.loss
        if phase == "warmup":
            return {"align": 1.0, "rel": 0.0, "analogy": 0.0, "disentangle": 0.0}
        elif phase == "relation":
            return {"align": 1.0, "rel": cfg.alpha_rel, "analogy": 0.0, "disentangle": 0.0}
        elif phase == "analogy":
            return {"align": 1.0, "rel": cfg.alpha_rel,
                    "analogy": cfg.beta_analogy, "disentangle": cfg.gamma_disentangle}
        else:
            return {"align": 0.5, "rel": cfg.alpha_rel,
                    "analogy": cfg.beta_analogy, "disentangle": cfg.gamma_disentangle}

    def train_epoch(self, scene_loader, analogy_loader=None):
        """Train for one epoch."""
        self.model.train()
        phase = self._get_phase(self.current_epoch)
        loss_weights = self._get_active_losses(phase)

        metrics = defaultdict(float)
        n_batches = 0
        accum = self.cfg.train.gradient_accumulation_steps
        self.optimizer.zero_grad()

        analogy_iter = iter(analogy_loader) if analogy_loader else None
        pbar = tqdm(scene_loader, desc=f"Epoch {self.current_epoch:3d} [{phase:8s}]")

        for batch_idx, batch in enumerate(pbar):
            total_loss = 0.0

            with autocast(enabled=self.cfg.train.use_amp):
                loss, batch_metrics = self._scene_step(batch, loss_weights)
                total_loss += loss
                for k, v in batch_metrics.items():
                    metrics[k] += v

                # Interleave analogy samples
                if analogy_iter and loss_weights.get("analogy", 0) > 0:
                    try:
                        a_batch = next(analogy_iter)
                        a_loss, a_metrics = self._analogy_step(a_batch, loss_weights)
                        total_loss += a_loss
                        for k, v in a_metrics.items():
                            metrics[k] += v
                    except StopIteration:
                        pass

            total_loss = total_loss / accum
            self.scaler.scale(total_loss).backward()

            n_batches += 1
            if (batch_idx + 1) % accum == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Log to TensorBoard (per optimizer step)
                if self.global_step % 10 == 0:
                    avg_loss = total_loss.item()
                    self.writer.add_scalar("train/loss_total", avg_loss, self.global_step)
                    self.writer.add_scalar("train/grad_norm", grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, self.global_step)
                    if "loss_align" in metrics:
                        self.writer.add_scalar("train/loss_align", metrics["loss_align"] / n_batches, self.global_step)
                    if "loss_rel" in metrics:
                        self.writer.add_scalar("train/loss_rel", metrics["loss_rel"] / n_batches, self.global_step)
                    if "loss_analogy" in metrics:
                        self.writer.add_scalar("train/loss_analogy", metrics["loss_analogy"] / n_batches, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                    self.writer.add_scalar("train/phase", ["warmup", "relation", "analogy", "finetune"].index(phase), self.global_step)

                self.global_step += 1

            # Progress bar update
            if (batch_idx + 1) % self.cfg.train.log_interval == 0:
                avg_metrics = {k: v / n_batches for k, v in metrics.items()}
                pbar.set_postfix(avg_metrics)

        # NaN detection
        for name, val in metrics.items():
            if val != val:  # NaN check
                print(f"\n[ALERT] NaN detected in {name} at epoch {self.current_epoch}!")
                self._log_alert("NaN_detected", {"metric": name, "epoch": self.current_epoch})

        for k in metrics:
            metrics[k] /= max(n_batches, 1)

        return dict(metrics)

    def validate(self, scene_loader, analogy_loader=None):
        """Validate on held-out data."""
        self.model.eval()
        metrics = defaultdict(float)
        n_batches = 0

        analogy_iter = iter(analogy_loader) if analogy_loader else None

        with torch.no_grad():
            for batch in scene_loader:
                batch = self._to_device(batch)
                z_v = self.model.encode_visual(batch["image"])
                z_s = self.model.encode_symbol(batch["tokens"])

                metrics["val_align"] += self.loss_align(z_v, z_s).item()

                # Relation loss on scene pairs (within-batch)
                if z_v.shape[0] >= 2:
                    mid = z_v.shape[0] // 2
                    if mid > 0:
                        za, zb = z_v[:mid], z_v[mid:2*mid]
                        sa, sb = z_s[:mid], z_s[mid:2*mid]
                        metrics["val_rel"] += self.loss_rel(za, zb, sa, sb).item()

                # Analogy evaluation (if analogy data available)
                if analogy_iter:
                    try:
                        a_batch = next(analogy_iter)
                        a_batch = self._to_device(a_batch)
                        za = self.model.encode_visual(a_batch["img_a"])
                        zb = self.model.encode_visual(a_batch["img_b"])
                        sa = self.model.encode_symbol(a_batch["sym_a"])
                        sb = self.model.encode_symbol(a_batch["sym_b"])
                        metrics["val_analogy"] += self.loss_analogy(za, zb, sa, sb).item()
                    except StopIteration:
                        pass

                n_batches += 1

        for k in metrics:
            metrics[k] /= max(n_batches, 1)
            self.writer.add_scalar(f"val/{k}", metrics[k], self.current_epoch)

        return dict(metrics)

    def _compute_disentangle(self, per_cat, tokens):
        """Compute disentanglement loss from per-category embeddings.

        Groups embeddings by attribute value within each category,
        then measures cross-category orthogonality.
        """
        attr_vectors = {}
        attr_categories = ["COL", "SIZE", "MAT"]  # disentangle non-identity attributes

        for cat in attr_categories:
            if cat not in per_cat or cat not in tokens:
                continue
            embeddings = per_cat[cat]          # (B, D)
            indices = tokens[cat]              # (B, N_objects)
            # Use first object's attribute value
            vals = indices[:, 0]               # (B,)

            # Group by attribute value and average
            unique_vals = vals.unique()
            cat_vectors = []
            for uv in unique_vals:
                mask = (vals == uv)
                if mask.sum() > 0:
                    cat_vectors.append(embeddings[mask].mean(dim=0))
            if len(cat_vectors) >= 2:
                attr_vectors[cat] = torch.stack(cat_vectors)  # (N_values, D)

        if len(attr_vectors) < 2:
            return torch.tensor(0.0, device=self.device)

        return self.loss_disentangle(attr_vectors)

    def _scene_step(self, batch, loss_weights):
        batch = self._to_device(batch)
        metrics = {}
        total_loss = 0.0

        z_v = self.model.encode_visual(batch["image"])
        z_s, per_cat = self.model.encode_symbol(batch["tokens"],
                                                 return_per_category=True)

        if loss_weights["align"] > 0:
            l_align = self.loss_align(z_v, z_s)
            total_loss += loss_weights["align"] * l_align
            metrics["loss_align"] = l_align.item()

        # Within-batch relational consistency
        if loss_weights["rel"] > 0 and z_v.shape[0] >= 2:
            mid = z_v.shape[0] // 2
            if mid > 0:
                z_va, z_vb = z_v[:mid], z_v[mid:2*mid]
                z_sa, z_sb = z_s[:mid], z_s[mid:2*mid]
                l_rel = self.loss_rel(z_va, z_vb, z_sa, z_sb)
                total_loss += loss_weights["rel"] * l_rel
                metrics["loss_rel"] = l_rel.item()

        # Disentanglement: per-category orthogonality
        if loss_weights.get("disentangle", 0) > 0 and per_cat:
            l_disent = self._compute_disentangle(per_cat, batch["tokens"])
            total_loss += loss_weights["disentangle"] * l_disent
            metrics["loss_disentangle"] = l_disent.item()

        return total_loss, metrics

    def _analogy_step(self, batch, loss_weights):
        batch = self._to_device(batch)
        metrics = {}
        total_loss = 0.0

        z_va = self.model.encode_visual(batch["img_a"])
        z_vb = self.model.encode_visual(batch["img_b"])
        z_sa, per_cat_a = self.model.encode_symbol(batch["sym_a"],
                                                     return_per_category=True)
        z_sb, per_cat_b = self.model.encode_symbol(batch["sym_b"],
                                                     return_per_category=True)

        if loss_weights["align"] > 0:
            l_align = (self.loss_align(z_va, z_sa) + self.loss_align(z_vb, z_sb)) / 2
            total_loss += loss_weights["align"] * l_align
            metrics["loss_align"] = l_align.item()

        if loss_weights["rel"] > 0:
            l_rel = self.loss_rel(z_va, z_vb, z_sa, z_sb)
            total_loss += loss_weights["rel"] * l_rel
            metrics["loss_rel"] = l_rel.item()

        if loss_weights["analogy"] > 0:
            l_analogy = self.loss_analogy(z_va, z_vb, z_sa, z_sb)
            total_loss += loss_weights["analogy"] * l_analogy
            metrics["loss_analogy"] = l_analogy.item()

        # Disentanglement on symbol embeddings
        if loss_weights.get("disentangle", 0) > 0 and per_cat_a:
            l_disent = (self._compute_disentangle(per_cat_a, batch["sym_a"]) +
                        self._compute_disentangle(per_cat_b, batch["sym_b"])) / 2
            total_loss += loss_weights["disentangle"] * l_disent
            metrics["loss_disentangle"] = l_disent.item()

        return total_loss, metrics

    def _to_device(self, batch):
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

    def _log_alert(self, alert_type: str, details: dict):
        """Log an alert to the training log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "alert",
            "alert": alert_type,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            **details,
        }
        with open(self.log_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    # ------------------------------------------------------------------
    #  Checkpoint & Resume
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str = None, metrics: dict = None,
                        is_best: bool = False):
        """Save a full training checkpoint."""
        if path is None:
            path = self.output_dir / f"checkpoint_epoch{self.current_epoch:03d}.pt"
        else:
            path = Path(path)

        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "metrics_history": dict(self.metrics_history),
            "patience_counter": self.patience_counter,
        }
        if metrics:
            ckpt["metrics"] = metrics

        torch.save(ckpt, str(path))

        # Track latest checkpoint (copy, no symlink needed on Windows)
        import shutil
        latest_path = self.output_dir / "checkpoint_latest.pt"
        shutil.copy2(str(path), str(latest_path))

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            shutil.copy2(str(path), str(best_path))

        # Log to structured log
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "checkpoint",
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "path": str(path),
            "is_best": is_best,
        }
        if metrics:
            entry["metrics"] = metrics
        with open(self.log_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

        return str(path)

    def load_checkpoint(self, path: str, resume_optimizer: bool = True):
        """Load a checkpoint and restore full training state.

        Args:
            path: checkpoint file path
            resume_optimizer: if True, restore optimizer, scheduler, and scaler state
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])

        if resume_optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.current_epoch = ckpt["epoch"]
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.patience_counter = ckpt.get("patience_counter", 0)

        if "metrics_history" in ckpt:
            self.metrics_history = defaultdict(list, ckpt["metrics_history"])

        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        print(f"  Best val loss so far: {self.best_val_loss:.4f}")
        return self.current_epoch

    def find_latest_checkpoint(self) -> str | None:
        """Find the latest checkpoint in the output directory."""
        # Check for symlink first
        latest_link = self.output_dir / "checkpoint_latest.pt"
        if latest_link.exists():
            return str(latest_link)

        # Search by pattern
        checkpoints = sorted(self.output_dir.glob("checkpoint_epoch*.pt"))
        if checkpoints:
            return str(checkpoints[-1])
        return None

    # ------------------------------------------------------------------
    #  Monitoring hooks
    # ------------------------------------------------------------------

    def check_improvement(self, val_loss: float) -> dict:
        """Check if validation loss improved and update best tracking.

        Returns a dict with monitoring signals.
        """
        signals = {"improved": False, "plateau_warning": False, "suggestion": None}

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            signals["improved"] = True
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience_limit:
                signals["plateau_warning"] = True
                signals["suggestion"] = (
                    f"No improvement for {self.patience_counter} epochs. "
                    "Consider reducing LR or checking for convergence."
                )

        return signals

    def close(self):
        """Clean up resources."""
        self.writer.close()

        # Write final summary
        summary = {
            "completed_at": datetime.now().isoformat(),
            "total_epochs": self.current_epoch,
            "total_steps": self.global_step,
            "best_val_loss": self.best_val_loss,
            "final_metrics": dict(self.metrics_history),
        }
        with open(self.output_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
