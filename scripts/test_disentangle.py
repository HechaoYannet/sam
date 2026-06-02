"""Quick test that disentangle computation works with per-category embeddings."""
import json, torch
from pathlib import Path
from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.trainer import SAMPipeline
from sam.data.dataset import build_vocabs, SceneDataset, AnalogyDataset, collate_fn
from torch.utils.data import DataLoader

cfg = Config()
cfg.device = 'cuda'
data_dir = Path('data/sam_dataset')
cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

with open(data_dir / 'scenes_train_meta.json') as f:
    scenes_meta = json.load(f)

scene_ds = SceneDataset(scenes_meta, cat_to_idx)
scene_loader = DataLoader(scene_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

visual = VisualEncoder(cfg.model, 256)
symbol = SymbolEncoder(cat_sizes, embed_dim=64, hidden_dim=128, manifold_dim=256)
model = SAMPipeline(visual, symbol, 256).cuda()

def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj

batch = next(iter(scene_loader))
batch = to_device(batch, 'cuda')

z_v = model.encode_visual(batch['image'])
z_s, per_cat = model.encode_symbol(batch['tokens'], return_per_category=True)
print(f'z_v: {z_v.shape}, z_s: {z_s.shape}')
print(f'Per-cat keys: {list(per_cat.keys())}')
for k, v in per_cat.items():
    print(f'  {k}: {v.shape}')

# Test trainer's _compute_disentangle
from sam.trainer import SAMTrainer
trainer = SAMTrainer(model, cfg, cat_sizes, output_dir='outputs/test_disentangle')

# Quick test with per_cat
l_disent = trainer._compute_disentangle(per_cat, batch['tokens'])
print(f'L_disentangle: {l_disent.item():.4f}')
print('Disentangle computation works!')
