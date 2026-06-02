import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """Cross-modal alignment loss.

    For paired (I, S) samples, visual and symbol embeddings should be close
    on the manifold. Uses both direct cosine loss and batch-level InfoNCE.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_v: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss.

        Args:
            z_v: (B, D) visual manifold embeddings (already normalized)
            z_s: (B, D) symbol manifold embeddings (already normalized)

        Returns:
            scalar loss
        """
        B = z_v.shape[0]

        # Direct cosine loss: paired samples should have cosine sim = 1
        cosine_sim = (z_v * z_s).sum(dim=-1)  # (B,)
        direct_loss = (1.0 - cosine_sim).mean()

        # InfoNCE contrastive loss
        logits = torch.matmul(z_v, z_s.T) / self.temperature  # (B, B)
        labels = torch.arange(B, device=z_v.device)

        loss_v2s = F.cross_entropy(logits, labels)
        loss_s2v = F.cross_entropy(logits.T, labels)

        infonce_loss = (loss_v2s + loss_s2v) / 2.0

        return direct_loss + 0.5 * infonce_loss
