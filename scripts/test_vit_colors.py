"""Test whether ViT-Tiny's raw features can distinguish colors.

Compares two ViT configs:
  1. Pretrained (ImageNet-1k) — as used in SAM
  2. Random init — baseline for what "no color knowledge" looks like
"""
import torch, numpy as np
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from sam.data.renderer import COLOR_MAP
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import timm

def render_square(rgb, size=224):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor((0.96,0.96,0.96))
    ax.add_patch(plt.Rectangle((-0.4,-0.4),0.8,0.8, facecolor=rgb))
    buf = BytesIO(); fig.savefig(buf,format='png',dpi=100,bbox_inches='tight',pad_inches=0)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert('RGB').resize((size,size), Image.BILINEAR)

transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

colors = ['red','blue','green','yellow','purple','orange']

for vit_name, pretrained in [("Random init", False), ("ImageNet-1k", True)]:
    print(f"\n{'='*50}")
    print(f"ViT-Tiny ({vit_name})")
    print(f"{'='*50}")

    if pretrained:
        import os
        vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
        local_dir = os.environ.get('SAM_PRETRAINED_VIT', '')
        if local_dir:
            import safetensors.torch
            wf = list(Path(local_dir).glob('*.safetensors'))[0]
            sd = safetensors.torch.load_file(str(wf))
            vit.load_state_dict(sd, strict=False)
    else:
        vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
    vit.eval()

    # Get embeddings for each color
    embeddings = {}
    for c in colors:
        embeds = []
        for _ in range(10):  # 10 trials with slight position jitter
            rng = np.random.RandomState(_)
            noise = rng.uniform(-0.005, 0.005, 3)
            rgb = tuple(np.clip(np.array(COLOR_MAP[c]) + noise, 0, 1))
            img = transform(render_square(rgb)).unsqueeze(0)
            with torch.no_grad():
                feat = vit.forward_features(img)[:, 0, :]  # CLS token
            embeds.append(feat)
        embeddings[c] = torch.cat(embeds, dim=0)  # (10, 192)

    # Intra-color consistency: avg cosine within same color
    print(f"\n{'Color':<10} {'Self-cos':<12} {'vs-green':<12} {'vs-red':<12}")
    print("-" * 46)
    for c in colors:
        e = embeddings[c]
        e_norm = e / e.norm(dim=1, keepdim=True)
        # Self-consistency: avg pairwise cosine within this color
        cos_mat = torch.matmul(e_norm, e_norm.T)
        n = cos_mat.shape[0]
        self_cos = (cos_mat.sum() - n) / (n * (n-1))  # exclude diagonal

        # Cross-color: cosine vs green and vs red
        g_norm = embeddings['green'] / embeddings['green'].norm(dim=1, keepdim=True)
        r_norm = embeddings['red'] / embeddings['red'].norm(dim=1, keepdim=True)
        vs_green = (e_norm @ g_norm.T).mean().item()
        vs_red = (e_norm @ r_norm.T).mean().item()

        print(f"{c:<10} {self_cos.item():.4f}       {vs_green:.4f}       {vs_red:.4f}")

    # Color separability: can we classify colors from ViT features?
    all_e = torch.cat([embeddings[c] for c in colors], dim=0)  # (60, 192)
    all_e = all_e / all_e.norm(dim=1, keepdim=True)
    labels = torch.cat([torch.full((10,), i) for i in range(6)])

    # Simple nearest-centroid accuracy
    centroids = torch.stack([embeddings[c].mean(0) for c in colors])
    centroids = centroids / centroids.norm(dim=1, keepdim=True)
    sim = torch.matmul(all_e, centroids.T)  # (60, 6)
    preds = sim.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    print(f"\n  Color classification accuracy (nearest centroid): {acc:.1%}")

print("\nConclusion: If pretrained ViT self-cos < 0.95 or classification < 80%,")
print("the color blindness is upstream of SAM training.")
