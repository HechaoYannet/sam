"""Color-balanced batch sampler for training.

Ensures each batch contains at least one sample of each color,
forcing the color contrastive loss to have meaningful positives.
"""

import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class ColorBalancedSampler(Sampler):
    """Sampler that draws batches with balanced color representation.

    Groups samples by color, then draws one sample from each color
    group per batch, filling the remainder uniformly.
    """

    def __init__(self, metadata: list, batch_size: int, shuffle: bool = True,
                 seed: int = 42):
        """
        Args:
            metadata: list of dicts, each must have a 'color' field
            batch_size: desired batch size (must be >= n_colors)
            shuffle: whether to shuffle between epochs
            seed: random seed
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        # Group indices by color
        self.color_groups = defaultdict(list)
        for i, item in enumerate(metadata):
            color = item.get("color", "unknown")
            self.color_groups[color].append(i)

        self.colors = sorted(self.color_groups.keys())
        self.n_colors = len(self.colors)

        if batch_size < self.n_colors:
            raise ValueError(f"batch_size ({batch_size}) must be >= "
                             f"n_colors ({self.n_colors})")

        self.n_total = len(metadata)
        # Each batch: 1 per color + (batch_size - n_colors) random
        self.n_batches = self.n_total // batch_size

    def __iter__(self):
        # Refill pools each epoch
        pools = {c: list(g) for c, g in self.color_groups.items()}
        if self.shuffle:
            for pool in pools.values():
                self.rng.shuffle(pool)

        batches = []
        for _ in range(self.n_batches):
            batch = []

            # Draw one from each color
            for color in self.colors:
                if not pools[color]:
                    # Refill if exhausted
                    pools[color] = list(self.color_groups[color])
                    if self.shuffle:
                        self.rng.shuffle(pools[color])
                batch.append(pools[color].pop())

            # Fill remaining slots from largest remaining pools
            all_remaining = []
            for pool in pools.values():
                all_remaining.extend(pool)

            if self.shuffle:
                self.rng.shuffle(all_remaining)

            n_needed = self.batch_size - len(batch)
            batch.extend(all_remaining[:n_needed])

            # Remove used items from pools
            used = set(all_remaining[:n_needed])
            for color in self.colors:
                pools[color] = [i for i in pools[color] if i not in used]

            if self.shuffle:
                self.rng.shuffle(batch)

            batches.append(batch)

        if self.shuffle:
            self.rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return self.n_batches
