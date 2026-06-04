import torch
import torch.nn as nn
import torch.nn.functional as F

from sam.encoders.color_encoder import ColorEncoder


class SymbolEncoder(nn.Module):
    """Structured symbol encoder with per-category embeddings and output heads.

    Each category gets its own dedicated output subspace (direct sum structure),
    preventing the MLP from freely mixing categories and collapsing dimensions.

    Output allocation (256 dims total):
      OBJ:  96  — shape identity (richest subspace)
      COL:  64  — color
      SIZE: 32  — size
      MAT:  32  — material
      REL:  16  — spatial relations
      PAD:  16  — padding token
    """

    # Per-category output allocation weights (relative, will be normalized to 256)
    _CATEGORY_WEIGHTS = {
        'OBJ': 12, 'COL': 8, 'SIZE': 4, 'MAT': 4, 'REL': 3, 'PAD': 1,
    }

    def __init__(self, cat_sizes: dict[str, int], embed_dim: int = 64,
                 hidden_dim: int = 512, manifold_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim

        # Per-category embedding tables
        self.embeddings = nn.ModuleDict()
        for cat, vocab_size in cat_sizes.items():
            self.embeddings[cat] = nn.Embedding(vocab_size, embed_dim,
                                                 padding_idx=vocab_size - 1)

        # Dynamically allocate output dims proportional to category weights
        cats_sorted = sorted(cat_sizes.keys())
        weights = [self._CATEGORY_WEIGHTS.get(c, 1) for c in cats_sorted]
        total_w = sum(weights)
        # Distribute manifold_dim proportionally, ensure each gets at least 8
        allocations = [max(8, int(manifold_dim * w / total_w)) for w in weights]
        # Adjust to exactly match manifold_dim
        diff = manifold_dim - sum(allocations)
        for i in range(abs(diff)):
            allocations[i % len(allocations)] += 1 if diff > 0 else -1

        self.heads = nn.ModuleDict()
        self._cat_allocations = {}
        for cat, alloc in zip(cats_sorted, allocations):
            self._cat_allocations[cat] = alloc
            self.heads[cat] = nn.Sequential(
                nn.Linear(embed_dim, alloc * 2),
                nn.LayerNorm(alloc * 2),
                nn.ReLU(),
                nn.Linear(alloc * 2, alloc),
                nn.LayerNorm(alloc),
            )

        # v8: Continuous color encoder (shared with discrete COL head)
        self.color_encoder = ColorEncoder(rgb_dim=3, embed_dim=embed_dim, hidden_dim=32)

        assert sum(allocations) == manifold_dim, \
            f"Allocation sum {sum(allocations)} != {manifold_dim}"

    def forward(self, tokens: dict[str, torch.Tensor],
                color_rgb: torch.Tensor | None = None,
                return_per_category: bool = False):
        """Encode structured symbol tokens to manifold point.

        Args:
            tokens: Dict mapping category -> LongTensor of shape (B, N_objects)
            color_rgb: Optional (B, 3) continuous RGB tensor for color path.
                       When provided, overrides the discrete COL token lookup.
            return_per_category: if True, also returns raw per-category embeddings

        Returns:
            z: (B, manifold_dim) manifold embeddings
            per_cat: (only if return_per_category=True) dict of per-category vectors
        """
        per_cat_raw = {}
        parts = []

        for cat in sorted(self.embeddings.keys()):
            head = self.heads[cat]
            emb_layer = self.embeddings[cat]

            if cat == 'COL' and color_rgb is not None:
                # Continuous color path: RGB → ColorEncoder → COL head
                cat_emb = self.color_encoder(color_rgb)
            elif cat in tokens:
                cat_indices = tokens[cat]
                cat_emb = emb_layer(cat_indices)
                if cat_emb.dim() == 3:  # (B, N_objects, D) → pool over objects
                    cat_emb = cat_emb.mean(dim=1)
                # else: (B, D) — already correct, no pooling needed
            else:
                # Missing category: use padding (last index)
                pad_idx = emb_layer.weight.shape[0] - 1
                cat_emb = emb_layer.weight[pad_idx].unsqueeze(0)

            per_cat_raw[cat] = cat_emb
            parts.append(head(cat_emb))

        z = torch.cat(parts, dim=-1)

        if return_per_category:
            return z, {cat: emb for cat, emb in per_cat_raw.items()}
        return z
