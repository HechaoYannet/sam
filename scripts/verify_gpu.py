"""GPU forward pass verification for P0."""
import torch, json
from pathlib import Path
from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline
from sam.data.dataset import build_vocabs, SceneDataset, collate_fn
from torch.utils.data import DataLoader


def to_device(obj, device):
    """Recursively move tensors to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


cfg = Config()
cfg.device = 'cuda'
cfg.model.vit_pretrained = False  # skip HF download for VRAM check only

cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

data_dir = Path('data/test_mini')
with open(data_dir / 'scenes_train_meta.json') as f:
    train_meta = json.load(f)

dataset = SceneDataset(train_meta, cat_to_idx)
loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

visual = VisualEncoder(cfg.model, manifold_dim=cfg.model.manifold_dim)
symbol = SymbolEncoder(cat_sizes, embed_dim=cfg.model.symbol_embed_dim,
                       hidden_dim=cfg.model.symbol_hidden_dim,
                       manifold_dim=cfg.model.manifold_dim)
model = SAMPipeline(visual, symbol, manifold_dim=cfg.model.manifold_dim)
model = model.cuda()

torch.cuda.reset_peak_memory_stats()
model.eval()
batch = next(iter(loader))
batch = to_device(batch, 'cuda')

with torch.no_grad():
    z_v = model.encode_visual(batch['image'])
    z_s = model.encode_symbol(batch['tokens'])

vram_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f'Forward pass: z_v={z_v.shape}, z_s={z_s.shape}')
print(f'Peak VRAM: {vram_mb:.1f} MB')
print(f'P0 threshold (4GB): {"PASS" if vram_mb < 4096 else "WARN"}')
print(f'Environment ready for P1 training.')
