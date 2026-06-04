"""Quick v8 evaluation — E1/E2/L1/manifold health."""
import sys, json, torch
from pathlib import Path
sys.path.insert(0, '.')
from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.manifold.projection import ManifoldProjection
from sam.data.dataset import build_vocabs, SceneDataset, collate_fn
from torch.utils.data import DataLoader
from scripts.comprehensive_eval import (
    evaluate_e1_retrieval, evaluate_e2_isomorphism,
    evaluate_e1_unseen_hues, compute_manifold_health,
)

cfg = Config()
cfg.device = 'cuda'
cfg.model.vit_pretrained = False
data_dir = Path('data/v8_dataset')
cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)

ckpt_path = Path('outputs/v8_full/checkpoint_epoch060.pt')
ckpt = torch.load(str(ckpt_path), map_location='cuda', weights_only=False)

visual = VisualEncoder(cfg.model, cfg.model.manifold_dim)
symbol = SymbolEncoder(cat_sizes, cfg.model.symbol_embed_dim,
                       manifold_dim=cfg.model.manifold_dim)

class EvalPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = visual
        self.symbol_encoder = symbol
        self.v_proj = ManifoldProjection(cfg.model.manifold_dim)
        self.s_proj = ManifoldProjection(cfg.model.manifold_dim)
    def encode_visual(self, x):
        return self.v_proj(self.visual_encoder(x))
    def encode_symbol(self, tokens, color_rgb=None, return_per_category=False,
                      return_pre_proj=False):
        if return_per_category or return_pre_proj:
            z_pre, per_cat = self.symbol_encoder(
                tokens, color_rgb=color_rgb, return_per_category=True)
            z_post = self.s_proj(z_pre)
            result = [z_post]
            if return_per_category: result.append(per_cat)
            if return_pre_proj: result.append(z_pre)
            return tuple(result) if len(result) > 1 else result[0]
        return self.s_proj(self.symbol_encoder(tokens, color_rgb=color_rgb))

model = EvalPipeline().to('cuda')
model_dict = model.state_dict()
matched = {k: v for k, v in ckpt['model_state_dict'].items()
           if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(matched)
model.load_state_dict(model_dict)
print(f'Loaded {len(matched)}/{len(model_dict)} params')
model.eval()

# Per-split evaluation
for split in ['train', 'val', 'test_iid', 'test_ood']:
    print(f'\n=== {split} ===')
    with open(data_dir / f'scenes_{split}_meta.json') as f:
        scenes = json.load(f)[:500]
    ds = SceneDataset(scenes, cat_to_idx)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    e1 = evaluate_e1_retrieval(model, loader, 'cuda')
    print(f'  E1 Top1={e1["e1_top1"]:.1%} Top3={e1["e1_top3"]:.1%} Top5={e1["e1_top5"]:.1%}')

    e2 = evaluate_e2_isomorphism(model, loader, 'cuda')
    print(f'  E2 Cos={e2["e2_mean_cosine"]:.4f} RSA={e2["e2_rsa_rho"]:.4f}')

# L1 unseen hues
print('\n=== L1 Unseen Hues ===')
try:
    l1 = evaluate_e1_unseen_hues(model, 'cuda')
    print(f'  Top1={l1["unseen_hue_top1"]:.1%} Top3={l1["unseen_hue_top3"]:.1%} Top5={l1["unseen_hue_top5"]:.1%}')
except Exception as ex:
    print(f'  Skipped: {ex}')

# Manifold health
print('\n=== Manifold Health ===')
health = compute_manifold_health(model, cat_to_idx, 'cuda')
for k in ['eff_rank', 'd50', 'd90', 'd95']:
    print(f'  {k}: {health[k]:.1f}')
for k in ['attr_OBJ_var', 'attr_COL_var', 'attr_SIZE_var', 'attr_MAT_var']:
    print(f'  {k}: {health[k]:.2%}')
for k in ['cross_COL_MAT_mean_abs_cos', 'cross_OBJ_SIZE_mean_abs_cos']:
    print(f'  {k}: {health[k]:.4f}')
print('\nDone!')
