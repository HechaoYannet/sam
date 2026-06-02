"""Attribute perturbation regularization.

Penalizes the "alignment shortcut" — when perfect cross-modal alignment
makes analogy degenerate to identity mapping. If we randomly perturb an
attribute in S_A and the model still correctly predicts S_B (via the
shortcut), we penalize it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerturbationLoss(nn.Module):
    """Penalizes models that use alignment shortcuts for analogy.

    During analogy training, with probability p, one attribute in S_A is
    replaced with a random value. If the model still produces the correct
    S_B embedding (high cosine) despite the perturbation, it's relying on
    alignment rather than attribute-level manipulation. This loss penalizes
    that behavior.
    """

    def __init__(self, perturb_prob: float = 0.3, margin: float = 0.5):
        super().__init__()
        self.perturb_prob = perturb_prob
        self.margin = margin

    def forward(self, z_vb, z_va, z_sa, z_sb_true, tokens_a,
                cat_sizes, rng=None):
        """Compute perturbation loss.

        Args:
            z_vb, z_va: (B, D) visual embeddings
            z_sa: (B, D) symbol embedding for scene A
            z_sb_true: (B, D) ground truth symbol embedding for scene B
            tokens_a: dict of {category: (B, N_objects)} token indices
            cat_sizes: dict of {category: vocab_size} for each category
            rng: numpy RandomState for reproducibility

        Returns:
            scalar loss
        """
        B = z_sa.shape[0]
        device = z_sa.device
        if rng is None:
            rng = __import__('numpy').random.RandomState(42)

        # Select which samples to perturb
        perturb_mask = torch.zeros(B, dtype=torch.bool, device=device)
        n_perturb = max(1, int(B * self.perturb_prob))
        perturb_idx = rng.choice(B, size=n_perturb, replace=False)
        perturb_mask[perturb_idx] = True

        if perturb_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Perturbable categories (non-REL, non-PAD)
        perturb_cats = ["COL", "SIZE", "MAT"]

        # For each perturbed sample, replace one attribute in the symbol
        perturbed_tokens = {}
        for cat in tokens_a:
            perturbed_tokens[cat] = tokens_a[cat].clone()

        for i in range(B):
            if not perturb_mask[i]:
                continue
            # Pick a random category and a random alternative value
            cat = perturb_cats[rng.randint(0, len(perturb_cats))]
            if cat not in cat_sizes:
                continue
            vocab_size = cat_sizes[cat]
            old_val = perturbed_tokens[cat][i, 0].item()
            # Pick a different value
            new_val = old_val
            attempts = 0
            while new_val == old_val and attempts < 20:
                new_val = rng.randint(0, vocab_size - 1)  # -1 for PAD
                attempts += 1
            perturbed_tokens[cat][i, 0] = new_val

        # We need to re-encode the perturbed symbols.
        # This is called from the trainer which has access to the model.
        # The loss returns the mask and perturbed tokens; actual encoding
        # is handled by the trainer.

        return perturb_mask, perturbed_tokens
