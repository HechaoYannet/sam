import torch
import torch.nn as nn
import torch.nn.functional as F


class SymbolEncoder(nn.Module):
    """Structured symbol encoder with per-category embeddings.

    Tokens for each category are embedded separately, then summed across
    categories and objects to produce a single manifold point.
    This design naturally supports compositionality: the representation
    of a scene is the sum of its attribute vectors.
    """

    def __init__(self, cat_sizes: dict[str, int], embed_dim: int = 64,
                 hidden_dim: int = 128, manifold_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim

        # Per-category embedding tables
        self.embeddings = nn.ModuleDict()
        for cat, vocab_size in cat_sizes.items():
            self.embeddings[cat] = nn.Embedding(vocab_size, embed_dim,
                                                 padding_idx=vocab_size - 1)

        total_embed_dim = len(cat_sizes) * embed_dim

        self.projection = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, manifold_dim),
            nn.LayerNorm(manifold_dim),
        )

    def forward(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode structured symbol tokens to manifold point.

        Args:
            tokens: Dict mapping category -> LongTensor of shape (B, N_objects)
                    where N_objects is 1 for single-object, 2 for scenes, etc.
                    REL category may have a different N.

        Returns:
            z: (B, manifold_dim) manifold embeddings
        """
        embeddings = []
        for cat, emb_layer in self.embeddings.items():
            if cat in tokens:
                cat_indices = tokens[cat]                     # (B, N_objects)
                cat_emb = emb_layer(cat_indices)               # (B, N_objects, embed_dim)
                cat_emb = cat_emb.mean(dim=1)                  # (B, embed_dim) — pool across objects
                embeddings.append(cat_emb)

        combined = torch.cat(embeddings, dim=-1)               # (B, total_embed_dim)
        z = self.projection(combined)                          # (B, manifold_dim)
        return z
