import torch
import torch.nn as nn
import torch.nn.functional as F


class ManifoldProjection(nn.Module):
    """Project and regularize encoder outputs onto the unit hypersphere."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.proj = nn.utils.parametrizations.orthogonal(
            nn.Linear(dim, dim)
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to unit hypersphere S^{d-1}.

        Args:
            x: (B, dim) raw encoder output

        Returns:
            z: (B, dim) normalized manifold point
        """
        z = self.ln(self.proj(x))
        z = F.normalize(z, p=2, dim=-1)
        return z
