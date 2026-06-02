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

    def forward(self, tokens: dict[str, torch.Tensor],
                return_per_category: bool = False):
        """Encode structured symbol tokens to manifold point.

        Args:
            tokens: Dict mapping category -> LongTensor of shape (B, N_objects)
            return_per_category: if True, also returns raw per-category embeddings
                as a dict of {category_name: (B, embed_dim)} for disentanglement.
                These are the pre-pooling raw embedding vectors (no extra parameters).

        Returns:
            z: (B, manifold_dim) manifold embeddings
            per_cat: (only if return_per_category=True) dict of per-category vectors
        """
        per_cat_raw = {}  # per-category embeddings before projection
        for cat, emb_layer in self.embeddings.items():
            if cat in tokens:
                cat_indices = tokens[cat]
                cat_emb = emb_layer(cat_indices)
                cat_emb = cat_emb.mean(dim=1)
                per_cat_raw[cat] = cat_emb

        # Build combined embedding
        combined_list = [per_cat_raw[cat] for cat in sorted(per_cat_raw.keys())]
        combined = torch.cat(combined_list, dim=-1)
        z = self.projection(combined)

        if return_per_category:
            # Return raw per-category embeddings (64-dim) for disentanglement.
            # No extra parameters — disentanglement works on raw embedding vectors.
            return z, {cat: emb for cat, emb in per_cat_raw.items()}
        return z
