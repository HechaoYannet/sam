import torch
import torch.nn as nn


class DisentangleLoss(nn.Module):
    """Encourage orthogonality between different attribute directions.

    For each attribute category (color, size, material, etc.), the average
    direction vectors across different values within that category should be
    orthogonal to directions from other categories.

    This pushes the manifold toward the direct-sum structure:
        M ≈ A_color ⊕ A_size ⊕ A_material ⊕ O ⊕ R
    """

    def forward(self, attribute_vectors: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute disentanglement loss.

        Args:
            attribute_vectors: Dict mapping category name -> (N_values, D) tensor
                              of average embeddings for each attribute value.

        Returns:
            scalar loss
        """
        # For each category, compute the subspace basis
        # We want cross-category direction vectors to be orthogonal
        cats = list(attribute_vectors.keys())
        if len(cats) < 2:
            return torch.tensor(0.0)

        loss = 0.0
        count = 0

        for i, cat_a in enumerate(cats):
            for cat_b in cats[i + 1:]:
                vecs_a = attribute_vectors[cat_a]  # (N_a, D)
                vecs_b = attribute_vectors[cat_b]  # (N_b, D)

                # Normalize
                vecs_a = torch.nn.functional.normalize(vecs_a, dim=-1)
                vecs_b = torch.nn.functional.normalize(vecs_b, dim=-1)

                # Cross-category absolute cosine similarity
                cross = torch.abs(torch.matmul(vecs_a, vecs_b.T))  # (N_a, N_b)
                loss += cross.mean()
                count += 1

        return loss / max(count, 1)
