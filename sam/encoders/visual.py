import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import timm


def _load_state_dict_from_path(path: str) -> dict:
    """Load state dict from a local file, supporting .safetensors and .pth."""
    path = Path(path)

    if path.suffix == ".safetensors":
        import safetensors.torch
        return safetensors.torch.load_file(str(path))
    elif path.suffix in (".pth", ".pt", ".bin"):
        # Try to find the right file in the directory
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            # Could be a full checkpoint or just state_dict
            if "state_dict" in ckpt:
                return ckpt["state_dict"]
            if "model_state_dict" in ckpt:
                return ckpt["model_state_dict"]
            return ckpt
        return {}
    else:
        raise ValueError(f"Unsupported weight format: {path.suffix}")


def _find_weight_file(directory: str) -> Optional[Path]:
    """Find the weight file in a directory (from ModelScope download)."""
    d = Path(directory)
    patterns = ["*.safetensors", "*.pth", "*.pt", "*.bin"]
    for pattern in patterns:
        files = list(d.glob(pattern))
        if files:
            return files[0]
    return None


class VisualEncoder(nn.Module):
    """ViT-Tiny visual encoder projecting images to the shared manifold.

    Supports loading pretrained weights from:
    - HuggingFace Hub (via timm built-in, requires network)
    - ModelScope local cache (via pretrained_path parameter)
    - Random initialization (pretrained=False, pretrained_path=None)
    """

    def __init__(self, cfg, manifold_dim: int = 256,
                 pretrained_path: Optional[str] = None):
        super().__init__()
        self.manifold_dim = manifold_dim

        # Try local path first, then env var, then fall back to HF
        use_pretrained = cfg.vit_pretrained
        local_weights = None

        if pretrained_path is None:
            pretrained_path = os.environ.get("SAM_PRETRAINED_VIT", None)

        if pretrained_path and os.path.exists(pretrained_path):
            path = Path(pretrained_path)
            if path.is_dir():
                weight_file = _find_weight_file(pretrained_path)
            else:
                weight_file = path

            if weight_file and weight_file.exists():
                local_weights = str(weight_file)
                use_pretrained = False  # load manually

        # Create model without pretrained if loading locally
        self.vit = timm.create_model(
            cfg.vit_model,
            pretrained=use_pretrained,
            num_classes=0,
        )

        # Load local weights if available
        if local_weights:
            print(f"Loading ViT weights from: {local_weights}")
            state_dict = _load_state_dict_from_path(local_weights)
            missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)} (head-related, expected)")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")

        vit_dim = self.vit.embed_dim  # 192 for vit_tiny

        # Freeze early layers
        self._freeze_layers(cfg.vit_freeze_layers)

        # Projection to manifold
        self.projection = nn.Sequential(
            nn.Linear(vit_dim, manifold_dim),
            nn.LayerNorm(manifold_dim),
        )

    def _freeze_layers(self, n_freeze: int):
        """Freeze the first n_freeze transformer blocks."""
        if n_freeze <= 0:
            return
        for name, param in self.vit.named_parameters():
            if "blocks" in name:
                block_idx = int(name.split(".")[1])
                if block_idx < n_freeze:
                    param.requires_grad = False
            elif "patch_embed" in name or "pos_drop" in name:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to manifold point.

        Args:
            x: (B, 3, H, W) RGB images

        Returns:
            z: (B, manifold_dim) manifold embeddings
        """
        features = self.vit.forward_features(x)
        cls_token = features[:, 0, :]
        z = self.projection(cls_token)
        return z
