"""Download ViT-Tiny pretrained weights from ModelScope.

Usage:
    conda run -n tct python scripts/download_vit_weights.py
    # Outputs the cache path for use in config
"""

import os
from pathlib import Path

from modelscope import snapshot_download

MODEL_ID = "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k"
CACHE_DIR = Path.home() / ".cache" / "sam" / "pretrained"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download():
    print(f"Downloading {MODEL_ID} from ModelScope...")
    local_dir = snapshot_download(
        MODEL_ID,
        cache_dir=str(CACHE_DIR),
    )
    local_dir = Path(local_dir)

    # Find the weight file
    weight_files = list(local_dir.glob("*.safetensors")) + \
                   list(local_dir.glob("*.pth")) + \
                   list(local_dir.glob("*.bin"))
    if weight_files:
        weight_path = weight_files[0]
        print(f"Weight file: {weight_path}")
        print(f"Size: {weight_path.stat().st_size / 1024**2:.1f} MB")
    else:
        print(f"Files in {local_dir}:")
        for f in sorted(local_dir.iterdir()):
            print(f"  {f.name}")
        # Try to find the pytorch_model.bin
        weight_path = local_dir / "pytorch_model.bin"

    print(f"\nCache directory: {local_dir}")
    print(f"\nSet this in your config or env:")
    print(f'  export SAM_PRETRAINED_VIT="{local_dir}"')
    return str(local_dir)


if __name__ == "__main__":
    download()
