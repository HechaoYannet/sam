import torch
import torch.nn as nn
import torch.nn.functional as F


class AnalogyLoss(nn.Module):
    """Analogy completion loss.

    Given (I_a, S_a, I_b), predicts S_b via manifold vector arithmetic:
        z_sb_pred = z_vb - z_va + z_sa

    The loss minimizes the distance between predicted and true S_b.
    """

    def forward(self, z_va: torch.Tensor, z_vb: torch.Tensor,
                z_sa: torch.Tensor, z_sb_true: torch.Tensor) -> torch.Tensor:
        """Compute analogy completion loss.

        Args:
            z_va: (B, D) visual embedding of scene A
            z_vb: (B, D) visual embedding of scene B
            z_sa: (B, D) symbol embedding of scene A
            z_sb_true: (B, D) symbol embedding of scene B (ground truth)

        Returns:
            scalar loss
        """
        # Vector arithmetic in the manifold
        z_sb_pred = z_vb - z_va + z_sa

        # Normalize prediction to avoid norm explosion
        z_sb_pred = torch.nn.functional.normalize(z_sb_pred, p=2, dim=-1, eps=1e-8)

        # Cosine distance between predicted and true
        cosine_sim = (z_sb_pred * z_sb_true).sum(dim=-1)
        # Clamp to [-1, 1] for numerical safety
        cosine_sim = cosine_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        loss = (1.0 - cosine_sim).mean()

        return loss
