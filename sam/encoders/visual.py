from typing import Optional

import torch
import torch.nn as nn
import timm


class VisualEncoder(nn.Module):
    """ViT-Tiny visual encoder projecting images to the shared manifold."""

    def __init__(self, cfg, manifold_dim: int = 256):
        super().__init__()
        self.manifold_dim = manifold_dim

        self.vit = timm.create_model(
            cfg.vit_model,
            pretrained=cfg.vit_pretrained,
            num_classes=0,  # remove classification head
        )
        vit_dim = self.vit.embed_dim  # 192 for vit_tiny

        # Freeze early layers
        self._freeze_layers(cfg.vit_freeze_layers)

        # Projection to manifold
        self.projection = nn.Sequential(
            nn.Linear(vit_dim, manifold_dim),
            nn.LayerNorm(manifold_dim),
        )

    def _freeze_layers(self, n_freeze: int):
        """Freeze the first n_freeze transformer blocks."""
        if n_freeze <= 0:
            return
        # ViT-Tiny has: patch_embed, pos_drop, blocks[0..11], norm, pre_logits
        # Freeze patch_embed and first n_freeze blocks
        for name, param in self.vit.named_parameters():
            if "blocks" in name:
                block_idx = int(name.split(".")[1])
                if block_idx < n_freeze:
                    param.requires_grad = False
            elif "patch_embed" in name or "pos_drop" in name:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to manifold point.

        Args:
            x: (B, 3, H, W) RGB images

        Returns:
            z: (B, manifold_dim) manifold embeddings
        """
        features = self.vit.forward_features(x)  # (B, 192)
        z = self.projection(features)             # (B, manifold_dim)
        return z
