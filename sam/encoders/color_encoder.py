"""Continuous RGB color encoder for SAM manifold.

Maps RGB color values to the same embedding space as discrete COL tokens,
sharing the same per-category output head.

Architecture: lightweight MLP (3 -> 32 -> 64) with LayerNorm.
This is intentionally small — the COL head (64->67) does the heavy lifting.
"""

import torch
import torch.nn as nn


class ColorEncoder(nn.Module):
    """Encode continuous RGB color to symbol embedding space.

    Input:  (B, 3) or (B, N_objects, 3) normalized RGB in [0, 1]
    Output: (B, embed_dim) or (B, N_objects, embed_dim)
    """

    def __init__(self, rgb_dim: int = 3, embed_dim: int = 64,
                 hidden_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(rgb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode RGB to embedding.

        Args:
            rgb: (..., 3) tensor with values in [0, 1]

        Returns:
            (..., embed_dim) tensor
        """
        return self.net(rgb)
