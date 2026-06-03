"""Test whether color recognition overfits to exact RGB values.

Generates three test conditions on the same shape (cube):
  A: Exact training RGB (from renderer's COLOR_MAP)
  B: Small perturbation (±5 RGB per channel)
  C: Medium perturbation (±15 RGB per channel)

If accuracy drops sharply from A→B→C, the model overfits to exact RGB.
"""
import torch, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline
from sam.data.dataset import build_vocabs, tokenize
from sam.data.renderer import COLOR_MAP
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

def render_solid_square(rgb, size=224):
    """Render a plain filled square with exact RGB color."""
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor((0.96, 0.96, 0.96))
    square = plt.Rectangle((-0.4, -0.4), 0.8, 0.8, facecolor=rgb)
    ax.add_patch(square)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB').resize((size, size), Image.BILINEAR)
    return img

cfg = Config(); cfg.device='cuda'; cfg.model.vit_pretrained=False
cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)
visual = VisualEncoder(cfg.model, 256)
symbol = SymbolEncoder(cat_sizes, 64, 128, 256)
model = SAMPipeline(visual, symbol, 256).cuda()
ckpt = torch.load('outputs/p2_wave1/checkpoint_best.pt', map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['model_state_dict']); model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

colors = ['red','blue','green','yellow','purple','orange']
rng = np.random.RandomState(42)

print(f"{'Color':<8} {'Exact RGB':<20} {'+/-5':<20} {'+/-15':<20}")
print("-" * 68)

for color_name in colors:
    base_rgb = COLOR_MAP[color_name]  # (R, G, B) in 0-1

    scores_exact = []
    scores_pert5 = []
    scores_pert15 = []

    for trial in range(10):  # 10 trials for statistical significance
        # Exact
        img_exact = render_solid_square(base_rgb)
        t_exact = transform(img_exact).unsqueeze(0).cuda()
        z_v = model.encode_visual(t_exact)
        best_cos = 0
        for c in colors:
            sym = f'[OBJ:cube] [COL:{c}] [SIZE:medium] [MAT:matte]'
            tok = tokenize(sym, cat_to_idx)
            tok = {k: v.unsqueeze(0).cuda() for k, v in tok.items()}
            z_s = model.encode_symbol(tok)
            cos = (z_v * z_s).sum().item()
            if c == color_name:
                best_cos = cos
        scores_exact.append(best_cos)

        # +/-5 perturbation
        noise = rng.uniform(-0.02, 0.02, size=3)  # ~5/255
        rgb5 = tuple(np.clip(np.array(base_rgb) + noise, 0, 1))
        img5 = render_solid_square(rgb5)
        t5 = transform(img5).unsqueeze(0).cuda()
        z_v5 = model.encode_visual(t5)
        for c in colors:
            sym = f'[OBJ:cube] [COL:{c}] [SIZE:medium] [MAT:matte]'
            tok = tokenize(sym, cat_to_idx)
            tok = {k: v.unsqueeze(0).cuda() for k, v in tok.items()}
            z_s = model.encode_symbol(tok)
            cos = (z_v5 * z_s).sum().item()
            if c == color_name:
                scores_pert5.append(cos)

        # +/-15 perturbation
        noise = rng.uniform(-0.06, 0.06, size=3)  # ~15/255
        rgb15 = tuple(np.clip(np.array(base_rgb) + noise, 0, 1))
        img15 = render_solid_square(rgb15)
        t15 = transform(img15).unsqueeze(0).cuda()
        z_v15 = model.encode_visual(t15)
        for c in colors:
            sym = f'[OBJ:cube] [COL:{c}] [SIZE:medium] [MAT:matte]'
            tok = tokenize(sym, cat_to_idx)
            tok = {k: v.unsqueeze(0).cuda() for k, v in tok.items()}
            z_s = model.encode_symbol(tok)
            cos = (z_v15 * z_s).sum().item()
            if c == color_name:
                scores_pert15.append(cos)

    mean_exact = np.mean(scores_exact)
    mean5 = np.mean(scores_pert5)
    mean15 = np.mean(scores_pert15)
    drop5 = (mean_exact - mean5) / mean_exact * 100
    drop15 = (mean_exact - mean15) / mean_exact * 100

    print(f"{color_name:<8} {mean_exact:.4f}             "
          f"{mean5:.4f} ({drop5:+.0f}%)      "
          f"{mean15:.4f} ({drop15:+.0f}%)")

print()
print("If ±5 drops > 10% or ±15 drops > 30%: CONFIRMED overfitting to exact RGB.")
