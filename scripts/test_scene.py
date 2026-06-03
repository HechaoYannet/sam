"""Test dual-object scene queries on a user photo."""
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

def encode_sym(model, sym_str, cat_to_idx, device):
    tok = tokenize(sym_str, cat_to_idx)
    tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
    return model.encode_symbol(tok)

cfg = Config()
cfg.device = 'cuda'; cfg.model.vit_pretrained = False
cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

visual = VisualEncoder(cfg.model, 256)
symbol = SymbolEncoder(cat_sizes, 64, 128, 256)
model = SAMPipeline(visual, symbol, 256).cuda()
ckpt = torch.load('outputs/p3/checkpoint_best.pt', map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
img = transform(Image.open('test/test.jpg').convert('RGB')).unsqueeze(0).cuda()

with torch.no_grad():
    z_v = model.encode_visual(img)

# Single-object queries
print('=== Single-object queries ===')
for label, col in [('left cube', 'blue'), ('right cube', 'orange')]:
    sym = f'[OBJ:cube] [COL:{col}] [SIZE:medium] [MAT:matte]'
    z_s = encode_sym(model, sym, cat_to_idx, 'cuda')
    cos = (z_v * z_s).sum().item()
    print(f'  {label} ({col}): cosine={cos:.4f}')

# Dual-object scene queries
print()
print('=== Dual-object scene queries ===')
scenes = [
    ('blue L of orange (CORRECT)',
     '[OBJ:cube] [COL:blue] [SIZE:medium] [MAT:matte] [REL:left_of] [OBJ:cube] [COL:orange] [SIZE:medium] [MAT:matte]'),
    ('orange R of blue',
     '[OBJ:cube] [COL:orange] [SIZE:medium] [MAT:matte] [REL:right_of] [OBJ:cube] [COL:blue] [SIZE:medium] [MAT:matte]'),
    ('blue R of orange (WRONG relation)',
     '[OBJ:cube] [COL:blue] [SIZE:medium] [MAT:matte] [REL:right_of] [OBJ:cube] [COL:orange] [SIZE:medium] [MAT:matte]'),
    ('spheres (WRONG shape)',
     '[OBJ:sphere] [COL:blue] [SIZE:medium] [MAT:matte] [REL:left_of] [OBJ:sphere] [COL:orange] [SIZE:medium] [MAT:matte]'),
]

for desc, sym_str in scenes:
    z_s = encode_sym(model, sym_str, cat_to_idx, 'cuda')
    cos = (z_v * z_s).sum().item()
    print(f'  {desc}: cosine={cos:.4f}')
