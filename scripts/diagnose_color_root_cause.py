"""Comprehensive color blindness root cause diagnosis.

Tests multiple hypotheses by tracing color signal through each layer:
  H2: L_disentangle competes with stronger losses (stuck near 0.5)
  H3: Color info is lost at a specific projection layer
  H4: ViT fine-tuning destroys color sensitivity (tested separately)

Note: H1 (ColorContrastiveLoss semantics) was tested in a previous iteration.
      The color_contrastive loss module has been removed; H1 is no longer tested here.
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
from sam.data.renderer import COLOR_MAP
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO


def render_solid_square(rgb, size=224):
    """Render a plain filled square with exact RGB color."""
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor((0.96, 0.96, 0.96))
    square = plt.Rectangle((-0.4, -0.4), 0.8, 0.8, facecolor=rgb)
    ax.add_patch(square)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB').resize((size, size), Image.BILINEAR)
    return img


class SAMPipelineStripped(torch.nn.Module):
    """SAMPipeline without tensorboard dependency."""

    def __init__(self, visual_encoder, symbol_encoder, manifold_dim=256):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.symbol_encoder = symbol_encoder
        self.v_proj = ManifoldProjection(manifold_dim)
        self.s_proj = ManifoldProjection(manifold_dim)

    def encode_visual(self, x, return_intermediates=False):
        """Encode with optional intermediate outputs for diagnosis."""
        # Step 1: ViT forward
        vit_features = self.visual_encoder.vit.forward_features(x)
        cls_token = vit_features[:, 0, :]  # (B, 192)

        # Step 2: VisualEncoder projection
        enc_proj = self.visual_encoder.projection(cls_token)  # (B, 256)

        # Step 3: Manifold projection
        z = self.v_proj(enc_proj)  # (B, 256) normalized

        if return_intermediates:
            return z, {
                "vit_cls": cls_token,
                "enc_projection": enc_proj,
                "manifold_out": z,
            }
        return z

    def encode_symbol(self, tokens, return_per_category=False):
        if return_per_category:
            z, per_cat = self.symbol_encoder(tokens, return_per_category=True)
            return self.s_proj(z), per_cat
        return self.s_proj(self.symbol_encoder(tokens))


def color_separability_metric(embeddings_dict):
    """Measure how well colors are separated in an embedding space.

    For each color, compute: (nearest other-color distance) / (self-consistency).
    Higher = better separation. Also compute nearest-centroid accuracy.

    Returns dict with metrics.
    """
    colors = sorted(embeddings_dict.keys())
    n_colors = len(colors)
    n_per_color = embeddings_dict[colors[0]].shape[0]
    dim = embeddings_dict[colors[0]].shape[1]

    # Compute centroids
    centroids = {}
    for c in colors:
        e = embeddings_dict[c]
        centroids[c] = e.mean(dim=0)

    # Normalize centroids
    centroid_mat = torch.stack([centroids[c] for c in colors])  # (6, D)
    centroid_mat = centroid_mat / centroid_mat.norm(dim=1, keepdim=True)

    # Nearest-centroid classification
    all_embeddings = []
    all_labels = []
    for i, c in enumerate(colors):
        all_embeddings.append(embeddings_dict[c])
        all_labels.append(torch.full((n_per_color,), i))

    all_e = torch.cat(all_embeddings, dim=0)
    all_e_norm = all_e / all_e.norm(dim=1, keepdim=True)
    all_labels = torch.cat(all_labels)

    sim = torch.matmul(all_e_norm, centroid_mat.T)
    preds = sim.argmax(dim=1)
    acc = (preds == all_labels).float().mean().item()

    # Cross-color cosine matrix
    cross_cos = torch.matmul(centroid_mat, centroid_mat.T)  # (6, 6)

    # Average cross-color cosine (excluding diagonal)
    mask = ~torch.eye(n_colors, dtype=torch.bool)
    avg_cross_cos = cross_cos[mask].mean().item()

    # Min cross-color cosine (most confused pair)
    min_cross = cross_cos[mask].min().item()

    # Intra-color consistency
    intra_cos = {}
    for c in colors:
        e = embeddings_dict[c]
        e_norm = e / e.norm(dim=1, keepdim=True)
        cos_mat = torch.matmul(e_norm, e_norm.T)
        n = cos_mat.shape[0]
        intra_cos[c] = (cos_mat.sum() - n) / (n * (n - 1))
    avg_intra = sum(intra_cos.values()) / len(intra_cos)

    return {
        "classification_acc": acc,
        "avg_cross_cosine": avg_cross_cos,
        "min_cross_cosine": min_cross,
        "avg_intra_cosine": avg_intra.item(),
        "separation_ratio": avg_intra.item() / max(avg_cross_cos, 1e-8),
        "per_color_intra": {c: v.item() for c, v in intra_cos.items()},
    }




def diagnose_layerwise_color_signal(model, transform, device):
    """H3: Trace color signal at each layer boundary."""
    print("\n" + "=" * 60)
    print("H3: Layer-wise color signal trace")
    print("=" * 60)

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    n_samples = 20

    # Collect embeddings at each layer
    layer_names = ["vit_cls", "enc_projection", "manifold_out"]
    layer_embeddings = {name: {} for name in layer_names}

    for c in colors:
        for name in layer_names:
            layer_embeddings[name][c] = []

    rng = np.random.RandomState(42)
    for c in colors:
        base_rgb = COLOR_MAP[c]
        for trial in range(n_samples):
            # Small perturbation for robustness
            noise = rng.uniform(-0.01, 0.01, 3)
            rgb = tuple(np.clip(np.array(base_rgb) + noise, 0, 1))
            img = render_solid_square(rgb)
            t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                _, intermediates = model.encode_visual(t,
                                                       return_intermediates=True)

            for name in layer_names:
                layer_embeddings[name][c].append(intermediates[name].cpu())

    # Stack per layer
    for name in layer_names:
        for c in colors:
            layer_embeddings[name][c] = torch.cat(layer_embeddings[name][c], dim=0)

    # Evaluate each layer
    for name in layer_names:
        metrics = color_separability_metric(layer_embeddings[name])
        print(f"\n--- {name} (dim={layer_embeddings[name][colors[0]].shape[1]}) ---")
        print(f"  Classification acc:  {metrics['classification_acc']:.1%}")
        print(f"  Avg cross-cos:       {metrics['avg_cross_cosine']:.4f}")
        print(f"  Min cross-cos:       {metrics['min_cross_cosine']:.4f} (most confused)")
        print(f"  Avg intra-cos:       {metrics['avg_intra_cosine']:.4f}")
        print(f"  Separation ratio:    {metrics['separation_ratio']:.2f}x")

    # Check where the drop happens
    vit_acc = color_separability_metric(layer_embeddings["vit_cls"])[
        "classification_acc"]
    manifold_acc = color_separability_metric(layer_embeddings["manifold_out"])[
        "classification_acc"]
    print(f"\n  ViT CLS → Manifold accuracy drop: {vit_acc:.1%} → {manifold_acc:.1%}")

    return layer_embeddings


def diagnose_disentanglement(model, cat_to_idx, device):
    """H2: Measure actual disentanglement of trained embeddings."""
    print("\n" + "=" * 60)
    print("H2: Embedding disentanglement measurement")
    print("=" * 60)

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    sizes = ['small', 'medium', 'large']
    materials = ['matte', 'shiny', 'metallic', 'glass']

    # Get symbol embeddings for all combinations of a single shape (cube)
    all_embeddings = {}
    for c in colors:
        for s in sizes:
            for m in materials:
                sym = f'[OBJ:cube] [COL:{c}] [SIZE:{s}] [MAT:{m}]'
                tok = tokenize(sym, cat_to_idx)
                tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
                with torch.no_grad():
                    z_s, per_cat = model.encode_symbol(tok,
                                                       return_per_category=True)
                all_embeddings[(c, s, m)] = {
                    "full": z_s.cpu(),
                    "per_cat": {k: v.cpu() for k, v in per_cat.items()},
                }

    # Measure disentanglement in per-category embedding space (64-dim)
    print("\nCross-category absolute cosine (per-category 64-dim embeddings):")
    attr_categories = ["COL", "SIZE", "MAT"]
    for cat_a in attr_categories:
        for cat_b in attr_categories:
            if cat_a >= cat_b:
                continue
            # Compute cross-category cosine matrix
            vecs_a = []
            vecs_b = []
            for (c, s, m), data in all_embeddings.items():
                vecs_a.append(data["per_cat"][cat_a])
                vecs_b.append(data["per_cat"][cat_b])

            vecs_a = torch.cat(vecs_a, dim=0)
            vecs_b = torch.cat(vecs_b, dim=0)
            vecs_a = vecs_a / vecs_a.norm(dim=1, keepdim=True)
            vecs_b = vecs_b / vecs_b.norm(dim=1, keepdim=True)

            cross = torch.abs(torch.matmul(vecs_a, vecs_b.T))
            mean_cross = cross.mean().item()
            max_cross = cross.max().item()
            print(f"  {cat_a} × {cat_b}: mean={mean_cross:.4f}, max={max_cross:.4f}")

    # Measure same-attribute cosine
    print("\nSame-attribute intra-value cosine (should be high):")
    for cat in attr_categories:
        if cat == "COL":
            values = colors
        elif cat == "SIZE":
            values = sizes
        else:
            values = materials

        for val in values:
            vecs = []
            for (c, s, m), data in all_embeddings.items():
                if (cat == "COL" and c == val) or \
                   (cat == "SIZE" and s == val) or \
                   (cat == "MAT" and m == val):
                    vecs.append(data["per_cat"][cat])
            vecs = torch.cat(vecs, dim=0)
            vecs = vecs / vecs.norm(dim=1, keepdim=True)
            intra = torch.matmul(vecs, vecs.T).mean().item()
            print(f"  {cat}={val}: intra-cos={intra:.4f}")

    # Check color subspace variance
    print("\nColor subspace analysis:")
    color_centroids = {}
    for c in colors:
        vecs = []
        for (cc, s, m), data in all_embeddings.items():
            if cc == c:
                vecs.append(data["per_cat"]["COL"])
        vecs = torch.cat(vecs, dim=0)
        color_centroids[c] = vecs.mean(dim=0, keepdim=True)
        color_centroids[c] = color_centroids[c] / color_centroids[c].norm(dim=1,
                                                                          keepdim=True)

    # Variance of color centroids
    all_color_c = torch.cat([color_centroids[c] for c in colors], dim=0)
    color_color_cross = torch.abs(torch.matmul(all_color_c, all_color_c.T))
    mask = ~torch.eye(len(colors), dtype=torch.bool)
    avg_color_cross = color_color_cross[mask].mean().item()
    print(f"  Mean cross-color centroid cosine: {avg_color_cross:.4f}")
    if avg_color_cross > 0.9:
        print("  >>> Colors are nearly IDENTICAL in embedding space (collapse!)")
    elif avg_color_cross > 0.7:
        print("  >>> Colors are highly correlated (poor separation)")
    elif avg_color_cross > 0.3:
        print("  >>> Colors moderately separated")
    else:
        print("  >>> Colors well separated")


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

    # Load Wave 1 checkpoint
    ckpt_path = Path("outputs/p2_wave1/checkpoint_best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("outputs/p3/checkpoint_best.pt")
        print(f"Wave 1 checkpoint not found, using P3: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"Epoch: {ckpt.get('epoch', 'unknown')}")

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Run diagnostics ---
    layer_data = diagnose_layerwise_color_signal(model, transform, device)
    diagnose_disentanglement(model, cat_to_idx, device)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    # Check layer-wise color drop
    vit_metrics = color_separability_metric(layer_data["vit_cls"])
    manifold_metrics = color_separability_metric(layer_data["manifold_out"])
    acc_drop = vit_metrics["classification_acc"] - manifold_metrics[
        "classification_acc"]
    if acc_drop > 0.3:
        print(f"H3 [CONFIRMED]: Color info lost in projection layers "
              f"({acc_drop:.1%} accuracy drop).")
    elif acc_drop > 0.05:
        print(f"H3 [PARTIAL]: Minor color info loss ({acc_drop:.1%} drop).")
    else:
        print(f"H3 [REJECTED]: No significant color loss in projections "
              f"({acc_drop:.1%} drop).")

    print("\nSee detailed output above for layer-wise metrics and H2 analysis.")


if __name__ == "__main__":
    main()
