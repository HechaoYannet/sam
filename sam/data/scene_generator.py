import colorsys
import copy
import itertools
import random
import os
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sam.config import DataConfig
from sam.data.renderer import ShapeRenderer


def _sample_hues(n: int, saturation: float = 0.8, value: float = 0.9,
                 offset: float = 0.0) -> list[tuple[float, float, float]]:
    """Sample n hues evenly on the HSV wheel.

    Args:
        n: number of hues
        saturation, value: HSV saturation and value (brightness)
        offset: hue offset in [0, 1] to shift the wheel

    Returns:
        List of (r, g, b) tuples with values in [0, 1]
    """
    hues = [(i / n + offset) % 1.0 for i in range(n)]
    colors = []
    for h in hues:
        rgb = colorsys.hsv_to_rgb(h, saturation, value)
        colors.append(tuple(rgb))
    return colors


def _build_attribute_combinations(cfg: DataConfig):
    """Build all possible single-object attribute combinations."""
    combos = []
    for obj in cfg.objects:
        for col in cfg.colors:
            for sz in cfg.sizes:
                for mat in cfg.materials:
                    combos.append({
                        "obj_type": obj,
                        "color": col,
                        "size": sz,
                        "material": mat,
                    })
    return combos


def _split_combinations(combos, cfg: DataConfig, rng: np.random.RandomState):
    """Split combinations for systematicity testing.

    Split by holding out attribute VALUE combinations, not instances.
    This tests whether the model can compose known primitives in unseen ways.
    """
    # Group combos by (color, size, material) — the composable attributes
    attr_groups = defaultdict(list)
    for c in combos:
        key = (c["color"], c["size"], c["material"])
        attr_groups[key].append(c)

    keys = list(attr_groups.keys())
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    n_test_iid = int(n * cfg.test_iid_ratio)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_iid_keys = set(keys[n_train + n_val:n_train + n_val + n_test_iid])
    test_ood_keys = set(keys[n_train + n_val + n_test_iid:])

    splits = {
        "train": [],
        "val": [],
        "test_iid": [],
        "test_ood": [],
    }

    for key, items in attr_groups.items():
        if key in train_keys:
            splits["train"].extend(items)
        elif key in val_keys:
            splits["val"].extend(items)
        elif key in test_iid_keys:
            splits["test_iid"].extend(items)
        else:
            splits["test_ood"].extend(items)

    return splits


def _obj_to_symbol_tokens(obj: dict) -> list[str]:
    """Convert object attributes to structured symbol tokens."""
    return [
        f"[OBJ:{obj['obj_type']}]",
        f"[COL:{obj['color']}]",
        f"[SIZE:{obj['size']}]",
        f"[MAT:{obj['material']}]",
    ]


def _obj_to_symbol_string(obj: dict) -> str:
    """Convert object attributes to a single symbol string."""
    return " ".join(_obj_to_symbol_tokens(obj))


def _scene_to_symbol_tokens(obj_a: dict, obj_b: dict, relation: str) -> list[str]:
    """Convert a full scene to structured symbol tokens."""
    return _obj_to_symbol_tokens(obj_a) + [f"[REL:{relation}]"] + _obj_to_symbol_tokens(obj_b)


def _scene_to_symbol_string(obj_a: dict, obj_b: dict, relation: str) -> str:
    """Convert a full scene to a single symbol string."""
    return " ".join(_scene_to_symbol_tokens(obj_a, obj_b, relation))


class SceneGenerator:
    """Generate synthetic 2D scenes with combinatorial attribute splits."""

    def __init__(self, cfg: DataConfig, output_dir: str, seed: int = 42):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.renderer = ShapeRenderer(image_size=cfg.image_size)
        self.rng = np.random.RandomState(seed)

        # Build attribute combinations
        self.all_combos = _build_attribute_combinations(cfg)
        self.combo_splits = _split_combinations(self.all_combos, cfg, self.rng)

        # Create output directories
        for split_name in ["train", "val", "test_iid", "test_ood"]:
            (self.output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (self.output_dir / "single_objects").mkdir(parents=True, exist_ok=True)

    def _sample_combo(self, split: str):
        return self.rng.choice(self.combo_splits[split]).copy()

    def generate_single_objects(self):
        """Generate all single-object images for warmup training."""
        print("Generating single-object images...")
        metadata = []

        for combo in tqdm(self.all_combos):
            for variant in range(self.cfg.single_object_variants):
                img = self.renderer.render_single_object(
                    obj_type=combo["obj_type"],
                    color_name=combo["color"],
                    size=combo["size"],
                    material=combo["material"],
                    angle_variant=variant,
                )
                fname = f"{combo['obj_type']}_{combo['color']}_{combo['size']}_{combo['material']}_v{variant}.png"
                img_path = self.output_dir / "single_objects" / fname
                img.save(img_path)

                metadata.append({
                    "image_path": str(img_path),
                    "symbol_tokens": " ".join(_obj_to_symbol_tokens(combo)),
                    "obj_type": combo["obj_type"],
                    "color": combo["color"],
                    "size": combo["size"],
                    "material": combo["material"],
                    "variant": variant,
                })

        print(f"Generated {len(metadata)} single-object images")
        return metadata

    def generate_scenes(self, n_scenes_per_split: dict = None):
        """Generate dual-object scene images.

        Args:
            n_scenes_per_split: dict mapping split name to count.
                Default: train=12000, val=3000, test_iid=3000, test_ood=3000
        """
        if n_scenes_per_split is None:
            n_scenes_per_split = {
                "train": 12000,
                "val": 3000,
                "test_iid": 3000,
                "test_ood": 3000,
            }

        all_metadata = {}

        for split_name, n_scenes in n_scenes_per_split.items():
            print(f"Generating {n_scenes} scenes for split '{split_name}'...")
            metadata = self._generate_split_scenes(split_name, n_scenes)
            all_metadata[split_name] = metadata
            print(f"  -> Generated {len(metadata)} scenes")

        return all_metadata

    def _generate_split_scenes(self, split_name: str, n_scenes: int):
        """Generate scenes for a specific split."""
        combos = self.combo_splits[split_name]
        relations = self.cfg.relations
        metadata = []

        if len(combos) < 2:
            print(f"  WARNING: split '{split_name}' has only {len(combos)} combos, skipping")
            return metadata

        for idx in tqdm(range(n_scenes)):
            # Sample two different objects
            idx_a, idx_b = self.rng.choice(len(combos), size=2, replace=False)
            obj_a = combos[idx_a].copy()
            obj_b = combos[idx_b].copy()
            relation = self.rng.choice(relations)

            img = self.renderer.render_scene(obj_a, obj_b, relation)
            fname = f"{split_name}_{idx:06d}.png"
            img_path = self.output_dir / "images" / split_name / fname
            img.save(img_path)

            metadata.append({
                "image_path": str(img_path),
                "scene_id": f"{split_name}_{idx:06d}",
                "obj_a": obj_a,
                "obj_b": obj_b,
                "relation": relation,
                "symbol_string": _scene_to_symbol_string(obj_a, obj_b, relation),
                "symbol_tokens": _scene_to_symbol_tokens(obj_a, obj_b, relation),
                "split": split_name,
            })

        return metadata

    def generate_analogy_samples(self, scenes_metadata, n_per_split: dict = None):
        """Generate analogy quadruplets from scene metadata.

        Each analogy sample: (img_A, sym_A, img_B) -> sym_B
        where img_A and img_B share some attributes but differ in others.
        """
        if n_per_split is None:
            n_per_split = {"train": 8000, "val": 2000, "test_iid": 2000, "test_ood": 2000}

        analogy_samples = {}

        for split_name, n_samples in n_per_split.items():
            split_scenes = scenes_metadata.get(split_name, [])
            if len(split_scenes) < 2:
                print(f"  WARNING: not enough scenes in '{split_name}' for analogy")
                continue

            print(f"Generating {n_samples} analogy samples for split '{split_name}'...")
            samples = []
            for _ in tqdm(range(n_samples)):
                idx_a, idx_b = self.rng.choice(len(split_scenes), size=2, replace=False)
                scene_a = split_scenes[idx_a]
                scene_b = split_scenes[idx_b]

                sample = {
                    "img_a_path": scene_a["image_path"],
                    "sym_a": scene_a["symbol_string"],
                    "img_b_path": scene_b["image_path"],
                    "sym_b": scene_b["symbol_string"],
                    "split": split_name,
                }
                samples.append(sample)

            analogy_samples[split_name] = samples
            print(f"  -> Generated {len(samples)} analogy samples")

        return analogy_samples

    def generate_structured_analogies(self, scenes_metadata,
                                      n_variants_per_scene: int = 5,
                                      n_per_split: dict = None):
        """Generate analogies where A and B differ by exactly ONE attribute.

        For each base scene, creates variants. Each variant changes one
        attribute of one object. The displacement vector encodes a single
        attribute change — making analogy learning tractable.

        Args:
            scenes_metadata: dict mapping split -> list of scene dicts
            n_variants_per_scene: max variants to create per scene
            n_per_split: dict mapping split -> max analogies to keep

        Returns:
            dict mapping split -> list of analogy samples
        """
        if n_per_split is None:
            n_per_split = {"train": 8000, "val": 2000,
                           "test_iid": 2000, "test_ood": 2000}

        attr_names = ["obj_type", "color", "size", "material", "relation"]
        alt_pools = {
            "obj_type": self.cfg.objects,
            "color": self.cfg.colors,
            "size": self.cfg.sizes,
            "material": self.cfg.materials,
            "relation": self.cfg.relations,
        }

        analogy_samples = {}

        for split_name, max_n in n_per_split.items():
            split_scenes = scenes_metadata.get(split_name, [])
            if len(split_scenes) < 2:
                print(f"  WARNING: not enough scenes in '{split_name}' for analogy")
                continue

            print(f"Generating structured analogies for '{split_name}'...")
            samples = []

            scenes_to_use = split_scenes[:max_n // max(n_variants_per_scene, 1) + 1]
            for scene in tqdm(scenes_to_use):
                for attr in attr_names[:n_variants_per_scene]:
                    variant = self._make_variant(scene, attr, alt_pools, split_name)
                    if variant is None:
                        continue

                    samples.append({
                        "img_a_path": scene["image_path"],
                        "sym_a": scene.get("symbol_string", ""),
                        "img_b_path": variant["image_path"],
                        "sym_b": variant["symbol_string"],
                        "changed_attribute": attr,
                        "split": split_name,
                    })

            # Trim to target count
            if len(samples) > max_n:
                indices = self.rng.choice(len(samples), size=max_n, replace=False)
                samples = [samples[i] for i in indices]

            analogy_samples[split_name] = samples
            print(f"  -> Generated {len(samples)} structured analogies")

        return analogy_samples

    def _make_variant(self, scene: dict, attr: str,
                      alt_pools: dict, split_name: str) -> dict | None:
        """Create a variant of a scene by changing one attribute.

        Renders the variant image and saves it to the images directory.
        """
        if attr == "relation":
            current_val = scene["relation"]
            alternatives = [r for r in alt_pools["relation"] if r != current_val]
            if not alternatives:
                return None
            new_rel = self.rng.choice(alternatives)
            variant_scene = copy.deepcopy(scene)
            variant_scene["relation"] = new_rel
            img = self.renderer.render_scene(
                variant_scene["obj_a"], variant_scene["obj_b"], new_rel)
        else:
            # Change attribute of a randomly chosen object
            obj_key = self.rng.choice(["obj_a", "obj_b"])
            obj = copy.deepcopy(scene[obj_key])
            current_val = obj.get(attr, "")
            alternatives = [v for v in alt_pools.get(attr, []) if v != current_val]
            if not alternatives:
                return None
            obj[attr] = self.rng.choice(alternatives)

            variant_scene = copy.deepcopy(scene)
            variant_scene[obj_key] = obj
            img = self.renderer.render_scene(
                variant_scene["obj_a"], variant_scene["obj_b"],
                variant_scene["relation"])

        # Save variant image
        fname = f"variant_{split_name}_{uuid.uuid4().hex}.png"
        img_path = self.output_dir / "images" / split_name / fname
        img.save(img_path)

        variant_scene["image_path"] = str(img_path)
        variant_scene["symbol_string"] = _scene_to_symbol_string(
            variant_scene["obj_a"], variant_scene["obj_b"],
            variant_scene["relation"])
        variant_scene["symbol_tokens"] = _scene_to_symbol_tokens(
            variant_scene["obj_a"], variant_scene["obj_b"],
            variant_scene["relation"])

        return variant_scene

    def generate_all(self, n_scenes_per_split=None, n_analogy_per_split=None):
        """Run the full data generation pipeline."""
        print("=" * 60)
        print("SAM Data Generation Pipeline")
        print("=" * 60)

        single_meta = self.generate_single_objects()
        scenes_meta = self.generate_scenes(n_scenes_per_split)
        if getattr(self.cfg, 'structured_analogy', False):
            analogy_meta = self.generate_structured_analogies(
                scenes_meta,
                n_variants_per_scene=self.cfg.n_analogy_variants_per_scene,
                n_per_split=n_analogy_per_split)
        else:
            analogy_meta = self.generate_analogy_samples(scenes_meta, n_analogy_per_split)

        # Compute summary statistics
        n_single = len(single_meta)
        n_scenes = sum(len(v) for v in scenes_meta.values())
        n_analogies = sum(len(v) for v in analogy_meta.values())

        print(f"\n{'=' * 60}")
        print(f"Generation complete:")
        print(f"  Single objects: {n_single}")
        print(f"  Dual-object scenes: {n_scenes}")
        print(f"  Analogy samples: {n_analogies}")
        print(f"  Total disk usage: todo")
        print(f"{'=' * 60}")

        return single_meta, scenes_meta, analogy_meta
