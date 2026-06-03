"""Symbol color separation loss.

Directly penalizes the symbol encoder's full output for collapsing color
information. For same-shape-different-color symbol pairs, penalizes high
cosine similarity. This forces the MLP projection to preserve color
distinctiveness in the final manifold embedding.
"""

import torch
import torch.nn as nn


class SymbolColorSeparationLoss(nn.Module):
    """Penalizes color collapse in the symbol encoder's full output.

    Per-category embeddings (64-dim) are already well disentangled by
    L_disentangle, but the MLP (384->hidden->256) can still ignore color.
    This loss directly supervises the final output to keep same-shape
    different-color embeddings apart.
    """

    def __init__(self, margin: float = 0.9):
        super().__init__()
        self.margin = margin

    def forward(self, z_s: torch.Tensor, shape_ids: torch.Tensor,
                color_ids: torch.Tensor) -> torch.Tensor:
        """Compute symbol color separation loss.

        Args:
            z_s: (B, D) full symbol manifold embeddings (normalized)
            shape_ids: (B,) shape category indices
            color_ids: (B,) color category indices

        Returns:
            scalar loss — mean violation above margin
        """
        B = z_s.shape[0]
        device = z_s.device

        # Mask: same shape, different color
        same_shape = shape_ids.unsqueeze(0) == shape_ids.unsqueeze(1)
        diff_color = color_ids.unsqueeze(0) != color_ids.unsqueeze(1)
        mask = same_shape & diff_color
        mask = mask.fill_diagonal_(False)

        n_pairs = mask.sum()
        if n_pairs == 0:
            return torch.tensor(0.0, device=device)

        # Cosine similarity matrix (z_s already normalized)
        cos = torch.matmul(z_s, z_s.T)

        # Penalize cosine above margin
        violation = torch.clamp(cos[mask] - self.margin, min=0)
        return violation.mean()
