"""Quick inference on a real image.

Usage:
    python scripts/infer.py --image my_photo.jpg --checkpoint outputs/p3/checkpoint_best.pt

    # Also specify symbol tokens manually:
    python scripts/infer.py --image my_photo.jpg --symbols "[OBJ:cube] [COL:red] [SIZE:medium] [MAT:matte]"
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline
from sam.data.dataset import build_vocabs, tokenize


def load_model(checkpoint_path, cfg, cat_sizes, device):
    visual = VisualEncoder(cfg.model, manifold_dim=cfg.model.manifold_dim)
    symbol = SymbolEncoder(cat_sizes, embed_dim=cfg.model.symbol_embed_dim,
                           hidden_dim=cfg.model.symbol_hidden_dim,
                           manifold_dim=cfg.model.manifold_dim)
    model = SAMPipeline(visual, symbol, manifold_dim=cfg.model.manifold_dim)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="outputs/p3/checkpoint_best.pt")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Symbol string, e.g. '[OBJ:cube] [COL:red] [SIZE:medium] [MAT:matte]'")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.model.vit_pretrained = False
    cat_to_idx, cat_sizes, idx_to_token = build_vocabs(cfg.data)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, cfg, cat_sizes, args.device)

    # Load and encode image
    img_tensor = preprocess_image(args.image).to(args.device)
    with torch.no_grad():
        z_v = model.encode_visual(img_tensor)  # (1, 256)

    # If symbol provided, compute alignment
    if args.symbols:
        tokens = tokenize(args.symbols, cat_to_idx)
        tokens = {k: v.unsqueeze(0).to(args.device) for k, v in tokens.items()}
        with torch.no_grad():
            z_s = model.encode_symbol(tokens)  # (1, 256)

        cosine = (z_v * z_s).sum(dim=-1).item()
        print(f"\nImage:  {args.image}")
        print(f"Symbol: {args.symbols}")
        print(f"Cosine similarity: {cosine:.4f}")
        print(f"Interpretation: {'Strong alignment' if cosine > 0.8 else 'Moderate' if cosine > 0.5 else 'Weak' if cosine > 0.2 else 'No alignment'}")

    # Enumerate all object+color combinations and find closest
    print(f"\nNearest object-color combinations:")
    all_combos = []
    all_z_s = []
    for obj in cfg.data.objects:
        for col in cfg.data.colors:
            sym_str = f"[OBJ:{obj}] [COL:{col}] [SIZE:medium] [MAT:matte]"
            tokens = tokenize(sym_str, cat_to_idx)
            tokens = {k: v.unsqueeze(0).to(args.device) for k, v in tokens.items()}
            with torch.no_grad():
                z_s = model.encode_symbol(tokens)
            all_combos.append(sym_str)
            all_z_s.append(z_s)

    all_z_s = torch.cat(all_z_s, dim=0)  # (30, 256)
    cosine_matrix = torch.matmul(z_v, all_z_s.T)  # (1, 30)
    top5_idx = cosine_matrix[0].topk(5).indices.tolist()

    for i, idx in enumerate(top5_idx):
        cos = cosine_matrix[0, idx].item()
        print(f"  #{i+1}: {all_combos[idx]}  (cos={cos:.4f})")

    # Displacement analysis: what attribute changes would the model predict?
    print(f"\nAttribute direction analysis (what the model 'sees'):")
    print(f"  Visual embedding norm: {z_v.norm(dim=-1).item():.4f}")
    print(f"  Embedding min/max/mean: {z_v.min().item():.4f} / {z_v.max().item():.4f} / {z_v.mean().item():.4f}")

    print(f"\nNote: Model was trained on synthetic 2D rendered shapes.")
    print(f"Real photos will show significant domain gap.")
    print(f"Low cosine similarity does NOT mean the model is broken —")
    print(f"it means the image is out of the training distribution.")


if __name__ == "__main__":
    main()
