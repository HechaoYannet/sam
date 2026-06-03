"""Test single-object recognition on user photos.

Usage:
    python scripts/test_single.py --image test/my_cube.jpg
    python scripts/test_single.py --image test/my_cube.jpg --ground_truth cube red
"""

import argparse, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline
from sam.data.dataset import build_vocabs, tokenize
from torchvision import transforms
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ground_truth", nargs=2, default=None, metavar=("SHAPE", "COLOR"),
                        help="e.g. 'cube red' — for comparison")
    parser.add_argument("--checkpoint", type=str, default="outputs/p3/checkpoint_best.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device; cfg.model.vit_pretrained = False
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

    # Load model
    visual = VisualEncoder(cfg.model, 256)
    symbol = SymbolEncoder(cat_sizes, 64, 128, 256)
    model = SAMPipeline(visual, symbol, 256).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load image
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(args.device)

    with torch.no_grad():
        z_v = model.encode_visual(img)

    # Enumerate all shape × color combinations
    results = []
    for obj in cfg.data.objects:
        for col in cfg.data.colors:
            sym = f"[OBJ:{obj}] [COL:{col}] [SIZE:medium] [MAT:matte]"
            tok = tokenize(sym, cat_to_idx)
            tok = {k: v.unsqueeze(0).to(args.device) for k, v in tok.items()}
            z_s = model.encode_symbol(tok)
            cos = (z_v * z_s).sum().item()
            results.append((obj, col, cos))

    results.sort(key=lambda x: -x[2])

    # Print results
    print(f"\nImage: {args.image}\n")
    print(f"{'Rank':<6}{'Shape':<12}{'Color':<10}{'Cosine':<10}")
    print("-" * 38)
    for i, (obj, col, cos) in enumerate(results[:10]):
        marker = ""
        if args.ground_truth and obj == args.ground_truth[0] and col == args.ground_truth[1]:
            marker = " ← GROUND TRUTH"
        print(f"{i+1:<6}{obj:<12}{col:<10}{cos:<10.4f}{marker}")

    # If ground truth specified, show its rank
    if args.ground_truth:
        gt_shape, gt_color = args.ground_truth
        for i, (obj, col, cos) in enumerate(results):
            if obj == gt_shape and col == gt_color:
                print(f"\nGround truth '{gt_shape} {gt_color}' rank: #{i+1}/30 (cos={cos:.4f})")
                break

    # Shape-level analysis
    print(f"\n--- Per-shape average cosine ---")
    shape_scores = {}
    for obj in cfg.data.objects:
        scores = [c for o, _, c in results if o == obj]
        shape_scores[obj] = sum(scores) / len(scores)
    for obj, score in sorted(shape_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 50) if score > 0 else ""
        print(f"  {obj:<12} {score:.4f} {bar}")

    # Color-level analysis
    print(f"\n--- Per-color average cosine ---")
    color_scores = {}
    for col in cfg.data.colors:
        scores = [c for _, co, c in results if co == col]
        color_scores[col] = sum(scores) / len(scores)
    for col, score in sorted(color_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 50) if score > 0 else ""
        print(f"  {col:<12} {score:.4f} {bar}")


if __name__ == "__main__":
    main()
