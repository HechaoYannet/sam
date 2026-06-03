"""Color-specific contrastive loss.

Forces the model to group same-color-different-shape samples together
in the manifold. This prevents shape from dominating and encourages
color-based organization. Same-color pairs are positives, pulling them
together across different shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorContrastiveLoss(nn.Module):
    """Within-batch contrastive loss that treats same-color-different-shape
    pairs as positives, pulling objects together by color and preventing
    shape from dominating the manifold."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, shape_ids: torch.Tensor,
                color_ids: torch.Tensor) -> torch.Tensor:
        """Compute color contrastive loss.

        Args:
            z: (B, D) manifold embeddings (already normalized)
            shape_ids: (B,) shape category indices (0-4 for 5 shapes)
            color_ids: (B,) color category indices (0-5 for 6 colors)

        Returns:
            scalar loss
        """
        B = z.shape[0]
        device = z.device

        # Cosine similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # (B, B)

        # Build positive mask: same color, DIFFERENT shape.
        # This pulls same-color objects together across shapes, encouraging
        # color-based organization instead of reinforcing shape dominance.
        same_color = color_ids.unsqueeze(0) == color_ids.unsqueeze(1)  # (B, B)
        diff_shape = shape_ids.unsqueeze(0) != shape_ids.unsqueeze(1)   # (B, B)
        pos_mask = same_color & diff_shape  # (B, B)

        # Self-exclusion
        pos_mask = pos_mask.fill_diagonal_(False)

        # If no positives in batch, return zero
        n_pos = pos_mask.sum()
        if n_pos == 0:
            return torch.tensor(0.0, device=device)

        # InfoNCE-style: each anchor predicts its positives
        loss = 0.0
        for i in range(B):
            positives = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(positives) == 0:
                continue

            # Log-sum-exp over all candidates (excluding self)
            neg_mask = torch.ones(B, dtype=torch.bool, device=device)
            neg_mask[i] = False  # exclude self
            lse = torch.logsumexp(sim[i][neg_mask], dim=0)

            # Average log-prob over positives
            pos_log_prob = sim[i][positives].mean() - lse
            loss -= pos_log_prob

        return loss / max(B, 1)
