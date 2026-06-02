import torch
import torch.nn as nn


class RelationalConsistencyLoss(nn.Module):
    """Enforce that displacement vectors are consistent across modalities.

    For two scenes (I_a, I_b) with symbol descriptions (S_a, S_b):
        || (v_a - v_b) - (s_a - s_b) ||²  should be minimized.

    This ensures that "what changes" between two scenes is represented by
    the same vector direction in both visual and symbol spaces.
    """

    def forward(self, z_va: torch.Tensor, z_vb: torch.Tensor,
                z_sa: torch.Tensor, z_sb: torch.Tensor) -> torch.Tensor:
        """Compute relational consistency loss.

        Args:
            z_va, z_vb: (B, D) visual manifold embeddings
            z_sa, z_sb: (B, D) symbol manifold embeddings

        Returns:
            scalar loss
        """
        d_v = z_va - z_vb  # visual displacement
        d_s = z_sa - z_sb  # symbol displacement

        # MSE between displacement vectors
        loss = ((d_v - d_s) ** 2).sum(dim=-1).mean()

        return loss
