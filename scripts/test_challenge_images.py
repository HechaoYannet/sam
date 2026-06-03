"""Test trained model on real challenge images from test/ directory."""
import torch, sys, json
from pathlib import Path
sys.path.insert(0, '.')
from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline
from sam.data.dataset import build_vocabs, tokenize
from torchvision import transforms
from PIL import Image

device = 'cuda'
cfg = Config(); cfg.model.vit_pretrained = False
cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

# Load model with widened MLP
visual = VisualEncoder(cfg.model, 256)
symbol = SymbolEncoder(cat_sizes, 64, 512, 256)
model = SAMPipeline(visual, symbol, 256).to(device)
ckpt = torch.load('outputs/p2_wave1_fix/checkpoint_epoch030.pt', map_location=device, weights_only=False)
md = model.state_dict()
pd = {k: v for k, v in ckpt['model_state_dict'].items() if k in md and md[k].shape == v.shape}
md.update(pd); model.load_state_dict(md); model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
shapes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']

def analyze_image(path, label=""):
    img = Image.open(path).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        z_v = model.encode_visual(t)

    # Test all (shape × color) combinations
    results = []
    for shape in shapes:
        for color in colors:
            sym = f'[OBJ:{shape}] [COL:{color}] [SIZE:medium] [MAT:matte]'
            tok = tokenize(sym, cat_to_idx)
            tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
            with torch.no_grad():
                z_s = model.encode_symbol(tok)
            cos = (z_v * z_s).sum().item()
            results.append((shape, color, cos))

    results.sort(key=lambda x: -x[2])

    # Also check different sizes and materials
    print(f"\n{'='*60}")
    print(f"Image: {label} ({path})")
    print(f"{'='*60}")
    print(f"Top matches:")
    for shape, color, cos in results[:5]:
        print(f"  {shape} {color}: cos={cos:.4f}")

    # Per-color breakdown for best shape
    best_shape = results[0][0]
    print(f"\nBest shape ({best_shape}) per color:")
    for color in colors:
        sym = f'[OBJ:{best_shape}] [COL:{color}] [SIZE:medium] [MAT:matte]'
        tok = tokenize(sym, cat_to_idx)
        tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
        with torch.no_grad():
            z_s = model.encode_symbol(tok)
        cos = (z_v * z_s).sum().item()
        print(f"  {color}: {cos:.4f}")

    # Cross-shape analysis: which color dominates?
    print(f"\nCross-shape color consistency (best color={results[0][1]}):")
    for shape in shapes:
        sym = f'[OBJ:{shape}] [COL:{results[0][1]}] [SIZE:medium] [MAT:matte]'
        tok = tokenize(sym, cat_to_idx)
        tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
        with torch.no_grad():
            z_s = model.encode_symbol(tok)
        cos = (z_v * z_s).sum().item()
        print(f"  {shape}: {cos:.4f}")

for path, label in [
    ('test/test_red_cube.png', 'Red Cube (render)'),
    ('test/test_blue_spuare.png', 'Blue Square (render)'),
    ('test/test_single.jpg', 'Single Object (photo)'),
    ('test/test.jpg', 'Scene (photo)'),
]:
    analyze_image(path, label)
