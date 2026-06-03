"""Diagnose shape-color interaction: why does color fail when shape is present?

Key question: Is the color blindness due to:
  A) Visual encoder can't separate colors of rendered shapes?
  B) Symbol encoder's FULL embedding is shape-dominated?
  C) Cross-modal matching fails (visual→symbol color mismatch)?
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.manifold.projection import ManifoldProjection
from sam.data.dataset import build_vocabs, tokenize
from sam.data.renderer import ShapeRenderer, COLOR_MAP
from torchvision import transforms
from PIL import Image


class SAMPipelineStripped(torch.nn.Module):
    def __init__(self, visual_encoder, symbol_encoder, manifold_dim=256):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.symbol_encoder = symbol_encoder
        self.v_proj = ManifoldProjection(manifold_dim)
        self.s_proj = ManifoldProjection(manifold_dim)

    def encode_visual(self, x):
        return self.v_proj(self.visual_encoder(x))

    def encode_symbol(self, tokens, return_per_category=False):
        if return_per_category:
            z, per_cat = self.symbol_encoder(tokens, return_per_category=True)
            return self.s_proj(z), per_cat
        return self.s_proj(self.symbol_encoder(tokens))

    def encode_symbol_raw(self, tokens):
        """Get symbol encoder output BEFORE manifold projection."""
        return self.symbol_encoder(tokens)


def color_separability(embeddings, labels, n_classes=6):
    """Nearest-centroid color classification accuracy."""
    # embeddings: (N, D), labels: (N,)
    centroids = []
    for i in range(n_classes):
        mask = labels == i
        c = embeddings[mask].mean(dim=0)
        c = c / c.norm()
        centroids.append(c)
    centroids = torch.stack(centroids)
    emb_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    sim = torch.matmul(emb_norm, centroids.T)
    preds = sim.argmax(dim=1)
    return (preds == labels).float().mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    cfg = Config()
    cfg.device = str(device)
    cfg.model.vit_pretrained = False
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

    visual = VisualEncoder(cfg.model, cfg.model.manifold_dim)
    symbol = SymbolEncoder(cat_sizes, cfg.model.symbol_embed_dim,
                           cfg.model.symbol_hidden_dim,
                           cfg.model.manifold_dim)
    model = SAMPipelineStripped(visual, symbol, cfg.model.manifold_dim).to(device)

    ckpt_path = Path("outputs/p2_wave1/checkpoint_best.pt")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    renderer = ShapeRenderer(image_size=224)
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']
    materials = ['matte']

    # =========================================================================
    # Test 1: Visual encoder — can it separate colors of RENDERED shapes?
    # =========================================================================
    print("=" * 60)
    print("Test 1: Visual encoder color separability on RENDERED shapes")
    print("=" * 60)

    for shape in shapes:
        z_v_list = []
        labels = []
        for ci, c in enumerate(colors):
            for trial in range(10):
                img = renderer.render_single_object(shape, c, 'medium',
                                                    'matte', angle_variant=trial % 3)
                t = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    z = model.encode_visual(t)
                z_v_list.append(z.cpu())
                labels.append(ci)

        z_v_all = torch.cat(z_v_list, dim=0)
        labels_t = torch.tensor(labels)
        acc = color_separability(z_v_all, labels_t)
        print(f"  {shape:<10}: color acc = {acc:.1%}")

    # =========================================================================
    # Test 2: Symbol encoder FULL output — color separability
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 2: Symbol encoder FULL output color separability")
    print("=" * 60)

    for shape in shapes:
        z_s_list = []
        labels = []
        for ci, c in enumerate(colors):
            sym = f'[OBJ:{shape}] [COL:{c}] [SIZE:medium] [MAT:matte]'
            tok = tokenize(sym, cat_to_idx)
            tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
            with torch.no_grad():
                z = model.encode_symbol(tok)
            z_s_list.append(z.cpu())
            labels.append(ci)

        z_s_all = torch.cat(z_s_list, dim=0)
        labels_t = torch.tensor(labels)
        acc = color_separability(z_s_all, labels_t)
        print(f"  {shape:<10}: color acc = {acc:.1%}")

    # =========================================================================
    # Test 3: Cross-modal color matching (the actual retrieval task)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 3: Cross-modal color retrieval (visual → symbol)")
    print("=" * 60)

    correct_top1 = 0
    correct_top3 = 0
    total = 0

    for shape in shapes:
        for true_c in colors:
            # Visual encoding of rendered shape
            img = renderer.render_single_object(shape, true_c, 'medium',
                                                'matte', angle_variant=0)
            t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                z_v = model.encode_visual(t)

            # Compare against all 6 color symbol embeddings
            scores = []
            for c in colors:
                sym = f'[OBJ:{shape}] [COL:{c}] [SIZE:medium] [MAT:matte]'
                tok = tokenize(sym, cat_to_idx)
                tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
                with torch.no_grad():
                    z_s = model.encode_symbol(tok)
                cos = (z_v * z_s).sum().item()
                scores.append((c, cos))

            scores.sort(key=lambda x: -x[1])
            rank = next(i + 1 for i, (c, _) in enumerate(scores) if c == true_c)
            top1 = scores[0][0] == true_c

            if top1:
                correct_top1 += 1
            if rank <= 3:
                correct_top3 += 1
            total += 1

            if not top1:
                print(f"  {shape} {true_c}: predicted {scores[0][0]} "
                      f"(cos={scores[0][1]:.3f}), true rank={rank}")

    print(f"\n  Top-1: {correct_top1}/{total} ({correct_top1/total:.1%})")
    print(f"  Top-3: {correct_top3}/{total} ({correct_top3/total:.1%})")

    # =========================================================================
    # Test 4: Per-shape breakdown
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 4: Per-shape color retrieval breakdown")
    print("=" * 60)

    for shape in shapes:
        correct = 0
        confusions = {}
        for true_c in colors:
            img = renderer.render_single_object(shape, true_c, 'medium',
                                                'matte', angle_variant=0)
            t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                z_v = model.encode_visual(t)

            scores = []
            for c in colors:
                sym = f'[OBJ:{shape}] [COL:{c}] [SIZE:medium] [MAT:matte]'
                tok = tokenize(sym, cat_to_idx)
                tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
                with torch.no_grad():
                    z_s = model.encode_symbol(tok)
                cos = (z_v * z_s).sum().item()
                scores.append((c, cos))

            scores.sort(key=lambda x: -x[1])
            top1 = scores[0][0]
            if top1 == true_c:
                correct += 1
            else:
                confusions[true_c] = top1

        print(f"  {shape:<10}: {correct}/{len(colors)} correct")
        if confusions:
            for tc, pred in confusions.items():
                print(f"    {tc} → {pred}")

    # =========================================================================
    # Test 5: Symbol embedding cosine matrix (are same-shape-diff-color
    #         symbol embeddings too similar?)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 5: Symbol embedding cosine matrix (cube, all colors)")
    print("=" * 60)

    cube_sym_embeddings = {}
    for c in colors:
        sym = f'[OBJ:cube] [COL:{c}] [SIZE:medium] [MAT:matte]'
        tok = tokenize(sym, cat_to_idx)
        tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
        with torch.no_grad():
            z = model.encode_symbol(tok)
        cube_sym_embeddings[c] = z.cpu()

    # Cosine matrix
    print(f"{'':<10}", end='')
    for c in colors:
        print(f"{c:<10}", end='')
    print()

    for c1 in colors:
        print(f"{c1:<10}", end='')
        for c2 in colors:
            cos = (cube_sym_embeddings[c1] * cube_sym_embeddings[c2]).sum().item()
            marker = " *" if c1 == c2 else ""
            print(f"{cos:.4f}{marker:<6}", end='')
        print()

    # Off-diagonal mean
    cos_off_diag = []
    for i, c1 in enumerate(colors):
        for j, c2 in enumerate(colors):
            if i != j:
                cos_off_diag.append(
                    (cube_sym_embeddings[c1] *
                     cube_sym_embeddings[c2]).sum().item())
    print(f"\n  Mean off-diagonal cosine: {np.mean(cos_off_diag):.4f}")
    if np.mean(cos_off_diag) > 0.95:
        print("  >>> SYMBOL EMBEDDINGS ARE NEARLY IDENTICAL for same-shape-diff-color!")
        print("  >>> Root cause: symbol encoder fails to distinguish colors for same shape.")

    # =========================================================================
    # Test 6: Visual embedding cosine matrix for rendered cubes
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 6: Visual embedding cosine matrix (rendered cubes, all colors)")
    print("=" * 60)

    cube_vis_embeddings = {}
    for c in colors:
        img = renderer.render_single_object('cube', c, 'medium', 'matte',
                                            angle_variant=0)
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            z = model.encode_visual(t)
        cube_vis_embeddings[c] = z.cpu()

    print(f"{'':<10}", end='')
    for c in colors:
        print(f"{c:<10}", end='')
    print()

    for c1 in colors:
        print(f"{c1:<10}", end='')
        for c2 in colors:
            cos = (cube_vis_embeddings[c1] * cube_vis_embeddings[c2]).sum().item()
            marker = " *" if c1 == c2 else ""
            print(f"{cos:.4f}{marker:<6}", end='')
        print()

    cos_off_diag_v = []
    for i, c1 in enumerate(colors):
        for j, c2 in enumerate(colors):
            if i != j:
                cos_off_diag_v.append(
                    (cube_vis_embeddings[c1] *
                     cube_vis_embeddings[c2]).sum().item())
    print(f"\n  Mean off-diagonal cosine: {np.mean(cos_off_diag_v):.4f}")


if __name__ == "__main__":
    main()
