"""VICReg-style global regularization + decoder bottleneck.

These losses replace per-attribute losses with scalable, attribute-agnostic
mechanisms that work for any number of attributes, continuous or discrete.

VICReg (Variance-Invariance-Covariance Regularization):
  - L_var: prevents dimension collapse (no dead dimensions)
  - L_cov: decorrelates dimensions (maximizes information per dimension)
  Both operate on the full embedding — no attribute list needed.

Decoder bottleneck:
  - Small MLP reconstructs all attributes from embedding
  - Information-theoretically forces all attribute info to be preserved
  - Scales to arbitrary attributes (just add decoder heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """Variance + Covariance regularization on manifold embeddings.

    Operates on the final manifold output. Attribute-agnostic — works
    for any number of attributes, discrete or continuous.
    """

    def __init__(self, var_weight: float = 0.5, cov_weight: float = 0.5,
                 gamma: float = 1.0):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute VICReg penalties.

        Args:
            z: (B, D) manifold embeddings (can be pre- or post-normalization)

        Returns:
            dict with 'var_loss', 'cov_loss', 'total_loss'
        """
        B, D = z.shape

        # Variance regularization: prevent dimension collapse
        # std(z, dim=0) measures how much each dimension varies across batch
        std_z = torch.sqrt(z.var(dim=0) + 1e-8)
        var_loss = F.relu(self.gamma - std_z).mean()

        # Covariance regularization: decorrelate dimensions
        # Off-diagonal elements of covariance matrix should be ~0
        z_centered = z - z.mean(dim=0)
        cov_z = (z_centered.T @ z_centered) / (B - 1)  # (D, D)
        # Zero out diagonal — we only penalize off-diagonal
        diag = torch.diag(cov_z)
        cov_loss = (cov_z ** 2).sum() - (diag ** 2).sum()
        cov_loss = cov_loss / D  # normalize by dimension count

        total = self.var_weight * var_loss + self.cov_weight * cov_loss
        return {"var_loss": var_loss, "cov_loss": cov_loss, "total_loss": total}


class DecoderBottleneck(nn.Module):
    """Small decoder that reconstructs attributes from manifold embedding.

    Information-theoretic guarantee: if the decoder can recover attributes
    from z, then z must contain all attribute information. No per-attribute
    loss design needed — just add a prediction head for each attribute.
    """

    def __init__(self, manifold_dim: int, cat_sizes: dict[str, int],
                 hidden_dim: int = 128):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.cat_sizes = cat_sizes

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Per-attribute prediction heads (lightweight)
        self.heads = nn.ModuleDict()
        for cat, vocab_size in cat_sizes.items():
            self.heads[cat] = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z: torch.Tensor,
                tokens: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict attributes from manifold embedding.

        Args:
            z: (B, manifold_dim) manifold embeddings
            tokens: {category: (B, N_objects)} token indices

        Returns:
            {'loss': scalar, 'loss_<cat>': scalar, 'acc_<cat>': scalar}
        """
        h = self.trunk(z)  # (B, hidden)
        results = {}
        total_loss = 0.0
        n_cats = 0

        for cat, head in self.heads.items():
            if cat not in tokens:
                continue
            labels = tokens[cat][:, 0]  # (B,) first object's attribute
            logits = head(h)            # (B, vocab)
            cat_loss = F.cross_entropy(logits, labels)
            total_loss = total_loss + cat_loss
            n_cats += 1

            # Accuracy for monitoring
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean()
            results[f"loss_{cat}"] = cat_loss
            results[f"acc_{cat}"] = acc

        results["loss"] = total_loss / max(n_cats, 1)
        return results


class ManifoldRegularizer(nn.Module):
    """Combined VICReg + DecoderBottleneck — the full scalable regularizer.

    Usage:
        reg = ManifoldRegularizer(manifold_dim, cat_sizes)
        vicreg_out = reg.vicreg(z_post)     # on final embeddings
        decoder_out = reg.decoder(z_post, tokens)  # attribute reconstruction
        total_reg = vicreg_out['total_loss'] + decoder_out['loss']
    """

    def __init__(self, manifold_dim: int, cat_sizes: dict[str, int],
                 var_weight: float = 0.5, cov_weight: float = 0.5,
                 decoder_hidden: int = 128):
        super().__init__()
        self.vicreg = VICRegLoss(var_weight=var_weight, cov_weight=cov_weight)
        self.decoder = DecoderBottleneck(manifold_dim, cat_sizes, decoder_hidden)

    def forward(self, z: torch.Tensor,
                tokens: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Full regularization forward pass.

        Args:
            z: (B, D) final manifold embeddings (post-projection)
            tokens: dict of token tensors

        Returns:
            combined metrics dict
        """
        vicreg_out = self.vicreg(z)
        decoder_out = self.decoder(z, tokens)
        return {**vicreg_out, **decoder_out}
