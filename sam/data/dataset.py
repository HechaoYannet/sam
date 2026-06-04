import json
import os
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def _parse_token(token: str) -> tuple[str, str]:
    """Parse a structured token like '[OBJ:cube]' into (category, value)."""
    inner = token.strip("[]")
    cat, val = inner.split(":", 1)
    return cat, val


def build_vocabs(cfg):
    """Build per-category vocabularies from DataConfig.

    Returns:
        cat_to_idx: dict mapping category name -> {value: idx}
        cat_sizes: dict mapping category name -> vocab size
        idx_to_token: dict mapping (category, idx) -> token string
    """
    cat_to_idx = {}
    cat_sizes = {}
    idx_to_token = {}

    categories = {
        "OBJ": cfg.objects,
        "COL": cfg.colors,
        "SIZE": cfg.sizes,
        "MAT": cfg.materials,
        "REL": cfg.relations,
    }

    for cat, values in categories.items():
        mapping = {v: i for i, v in enumerate(values)}
        mapping["[PAD]"] = len(values)  # padding token
        cat_to_idx[cat] = mapping
        cat_sizes[cat] = len(values) + 1  # +1 for PAD
        for v, i in mapping.items():
            idx_to_token[(cat, i)] = f"[{cat}:{v}]" if v != "[PAD]" else "[PAD]"

    return cat_to_idx, cat_sizes, idx_to_token


def tokenize(symbol_string: str, cat_to_idx: dict, max_objects: int = 2,
             color_rgb: torch.Tensor | None = None):
    """Convert a symbol string to per-category index tensors.

    Args:
        symbol_string: e.g. "[OBJ:cube] [COL:red] ... [REL:left_of] [OBJ:sphere] ..."
        cat_to_idx: category -> {value: idx} mapping
        color_rgb: optional (max_objects, 3) float tensor of continuous RGB values.
                   When provided, stored in the returned dict under the key "color_rgb".

    Returns:
        Dict mapping category -> LongTensor of indices.
        For two-object scenes, each category tensor has shape (2,) except REL which is (1,).
        Padded with [PAD] token when only one object (single-object mode).
    """
    tokens = symbol_string.split()
    parsed = [_parse_token(t) for t in tokens]

    # Group by category
    result = defaultdict(list)
    for cat, val in parsed:
        if cat == "REL":
            result["REL"].append(cat_to_idx["REL"].get(val, cat_to_idx["REL"]["[PAD]"]))
        else:
            result[cat].append(cat_to_idx[cat].get(val, cat_to_idx[cat]["[PAD]"]))

    # Convert to tensors
    output = {}
    for cat in ["OBJ", "COL", "SIZE", "MAT"]:
        indices = result.get(cat, [])
        # Pad to max_objects
        while len(indices) < max_objects:
            indices.append(cat_to_idx[cat]["[PAD]"])
        output[cat] = torch.tensor(indices[:max_objects], dtype=torch.long)

    for cat in ["REL"]:
        indices = result.get(cat, [cat_to_idx["REL"]["[PAD]"]] * max_objects)
        output[cat] = torch.tensor(indices[:max_objects], dtype=torch.long)

    if color_rgb is not None:
        output["color_rgb"] = color_rgb

    return output


class SingleObjectDataset(Dataset):
    """Dataset for single-object alignment training (warmup phase)."""

    def __init__(self, metadata: list, cat_to_idx: dict, image_size: int = 224):
        self.metadata = metadata
        self.cat_to_idx = cat_to_idx
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        img_tensor = self.transform(img)

        # For single objects, pad the symbol tokens to match the dual-object format
        tokens = {cat: torch.tensor([idx_val], dtype=torch.long)
                  for cat, idx_val in tokenize(item["symbol_tokens"], self.cat_to_idx, max_objects=1).items()}

        return {
            "image": img_tensor,
            "tokens": tokens,
            "symbol_string": item["symbol_tokens"],
        }


class SceneDataset(Dataset):
    """Dataset for dual-object scene training."""

    def __init__(self, metadata: list, cat_to_idx: dict,
                 transform=None, image_size: int = 224):
        self.metadata = metadata
        self.cat_to_idx = cat_to_idx
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        img_tensor = self.transform(img)

        # Check for continuous color metadata
        color_rgb = None
        if item.get("color_type") == "continuous":
            rgb_a = torch.tensor(
                item["obj_a"].get("color_rgb", (0.5, 0.5, 0.5)),
                dtype=torch.float32)
            rgb_b = torch.tensor(
                item["obj_b"].get("color_rgb", (0.5, 0.5, 0.5)),
                dtype=torch.float32)
            color_rgb = torch.stack([rgb_a, rgb_b])  # (2, 3)

        tokens = tokenize(item["symbol_string"], self.cat_to_idx,
                         color_rgb=color_rgb)

        result = {
            "image": img_tensor,
            "tokens": tokens,
            "symbol_string": item["symbol_string"],
            "relation": item["relation"],
        }
        if color_rgb is not None:
            result["color_rgb"] = color_rgb
        return result


class AnalogyDataset(Dataset):
    """Dataset for analogy completion training.

    Each sample: (img_A, sym_A, img_B) -> sym_B
    """

    def __init__(self, samples: list, cat_to_idx: dict, image_size: int = 224):
        self.samples = samples
        self.cat_to_idx = cat_to_idx
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_a = Image.open(sample["img_a_path"]).convert("RGB")
        img_b = Image.open(sample["img_b_path"]).convert("RGB")

        img_a_tensor = self.transform(img_a)
        img_b_tensor = self.transform(img_b)

        # Check for continuous color in analogy samples
        color_rgb_a = None
        color_rgb_b = None

        if sample.get("color_rgb_a") is not None:
            color_rgb_a = torch.tensor(sample["color_rgb_a"], dtype=torch.float32)
        if sample.get("color_rgb_b") is not None:
            color_rgb_b = torch.tensor(sample["color_rgb_b"], dtype=torch.float32)

        sym_a_tokens = tokenize(sample["sym_a"], self.cat_to_idx,
                               color_rgb=color_rgb_a)
        sym_b_tokens = tokenize(sample["sym_b"], self.cat_to_idx,
                               color_rgb=color_rgb_b)

        result = {
            "img_a": img_a_tensor,
            "img_b": img_b_tensor,
            "sym_a": sym_a_tokens,
            "sym_b": sym_b_tokens,
        }
        if color_rgb_a is not None:
            result["color_rgb_a"] = color_rgb_a
        if color_rgb_b is not None:
            result["color_rgb_b"] = color_rgb_b
        result["changed_attribute"] = sample.get("changed_attribute", "unknown")
        return result


def collate_fn(batch):
    """Collate function for DataLoader that handles token dicts."""
    result = {}
    for key in batch[0].keys():
        if key in ("tokens", "sym_a", "sym_b"):
            # Merge token dicts into batched tensors
            token_dicts = [b[key] for b in batch]
            merged = {}
            for cat in token_dicts[0].keys():
                merged[cat] = torch.stack([td[cat] for td in token_dicts])
            result[key] = merged
        elif key in ("image", "img_a", "img_b"):
            result[key] = torch.stack([b[key] for b in batch])
        elif key in ("color_rgb", "color_rgb_a", "color_rgb_b"):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    return result
