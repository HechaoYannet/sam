"""Test color recognition on synthetic in-distribution data."""
import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline
from sam.data.dataset import build_vocabs, tokenize
from torchvision import transforms
from PIL import Image

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

# Test all cubes in all colors
print(f"{'True':<8}", end='')
for i in range(6):
    print(f"{'#'+str(i+1):<22}", end='')
print()

for true_col in colors:
    path = f'data/test_mini/single_objects/cube_{true_col}_medium_matte_v0.png'
    img = transform(Image.open(path).convert('RGB')).unsqueeze(0).cuda()
    with torch.no_grad():
        z_v = model.encode_visual(img)

    scores = []
    for c in colors:
        sym = f'[OBJ:cube] [COL:{c}] [SIZE:medium] [MAT:matte]'
        tok = tokenize(sym, cat_to_idx)
        tok = {k: v.unsqueeze(0).cuda() for k, v in tok.items()}
        z_s = model.encode_symbol(tok)
        cos = (z_v * z_s).sum().item()
        scores.append((c, cos))
    scores.sort(key=lambda x: -x[1])

    print(f'{true_col:<8}', end='')
    for rank, (pred_col, cos) in enumerate(scores):
        marker = ' ✓' if pred_col == true_col else ''
        print(f'{pred_col} {cos:.3f}{marker:<6}', end='')
    print()

# Summary
print(f"\n{'Color':<10} {'Rank':<6} {'Top-1?'}")
print("-" * 30)
correct = 0
for true_col in colors:
    path = f'data/test_mini/single_objects/cube_{true_col}_medium_matte_v0.png'
    img = transform(Image.open(path).convert('RGB')).unsqueeze(0).cuda()
    with torch.no_grad():
        z_v = model.encode_visual(img)

    scores = []
    for c in colors:
        sym = f'[OBJ:cube] [COL:{c}] [SIZE:medium] [MAT:matte]'
        tok = tokenize(sym, cat_to_idx)
        tok = {k: v.unsqueeze(0).cuda() for k, v in tok.items()}
        z_s = model.encode_symbol(tok)
        cos = (z_v * z_s).sum().item()
        scores.append((c, cos))
    scores.sort(key=lambda x: -x[1])

    rank = next(i+1 for i, (c, _) in enumerate(scores) if c == true_col)
    top1 = scores[0][0]
    is_correct = top1 == true_col
    if is_correct: correct += 1
    print(f'{true_col:<10} #{rank:<5} {"✓" if is_correct else "✗ (got " + top1 + ")"}')

print(f'\nAccuracy: {correct}/{len(colors)}')
