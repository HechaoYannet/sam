"""SAM Data Generation Script.

Generates the full CausalFlow-Synth-MM dataset:
  - Single-object images (~1000)
  - Dual-object scene images (~20,000)
  - Analogy quadruplets (~14,000)

Usage:
    python scripts/generate_data.py --output_dir data/sam_dataset --seed 42
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config
from sam.data.scene_generator import SceneGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate SAM dataset")
    parser.add_argument("--output_dir", type=str, default="data/sam_dataset",
                        help="Output directory for generated data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n_train", type=int, default=12000,
                        help="Number of training scenes")
    parser.add_argument("--n_val", type=int, default=3000,
                        help="Number of validation scenes")
    parser.add_argument("--n_test_iid", type=int, default=3000,
                        help="Number of IID test scenes")
    parser.add_argument("--n_test_ood", type=int, default=3000,
                        help="Number of OOD test scenes")
    parser.add_argument("--n_analogy_train", type=int, default=8000,
                        help="Number of training analogy samples")
    parser.add_argument("--n_analogy_val", type=int, default=2000,
                        help="Number of validation analogy samples")
    parser.add_argument("--n_analogy_test", type=int, default=4000,
                        help="Number of test analogy samples")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed

    output_dir = Path(args.output_dir)

    # Initialize generator
    generator = SceneGenerator(
        cfg=cfg.data,
        output_dir=str(output_dir),
        seed=args.seed,
    )

    # Generate all data
    single_meta, scenes_meta, analogy_meta = generator.generate_all(
        n_scenes_per_split={
            "train": args.n_train,
            "val": args.n_val,
            "test_iid": args.n_test_iid,
            "test_ood": args.n_test_ood,
        },
        n_analogy_per_split={
            "train": args.n_analogy_train,
            "val": args.n_analogy_val,
            "test_iid": args.n_analogy_test // 2,
            "test_ood": args.n_analogy_test // 2,
        },
    )

    # Save metadata
    def _serialize(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

    with open(output_dir / "single_objects_meta.json", "w") as f:
        json.dump(single_meta, f, indent=2, default=_serialize)

    for split_name, meta in scenes_meta.items():
        with open(output_dir / f"scenes_{split_name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=_serialize)

    for split_name, meta in analogy_meta.items():
        with open(output_dir / f"analogy_{split_name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=_serialize)

    # Summary
    n_single = len(single_meta)
    n_scenes = sum(len(v) for v in scenes_meta.values())
    n_analogies = sum(len(v) for v in analogy_meta.values())

    print(f"\nDone! Dataset saved to {output_dir}")
    print(f"  Single-object images: {n_single}")
    print(f"  Dual-object scenes:   {n_scenes}")
    print(f"  Analogy samples:      {n_analogies}")


if __name__ == "__main__":
    main()
