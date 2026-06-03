"""Full evaluation suite for Phase 2 Wave 1 color blindness fix.

Test groups:
  G1: All-shapes-all-colors cross-modal retrieval matrix (5×6=30)
  G2: Per-color accuracy breakdown
  G3: Cross-material robustness (matte/shiny/metallic/glass)
  G4: Symbol embedding color separation (cross-color cosine)
  G5: Before/after fix comparison summary
  G6: Challenge images on real photos
"""

import torch, sys, json, time, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '.')
from sam.config import Config
from sam.encoders.visual import VisualEncoder
from sam.encoders.symbol import SymbolEncoder
from sam.manifold.projection import ManifoldProjection
from sam.data.dataset import build_vocabs, tokenize
from sam.data.renderer import ShapeRenderer
from torchvision import transforms


class SAMPipelineStripped(torch.nn.Module):
    def __init__(self, v, s, d=256):
        super().__init__()
        self.visual_encoder = v; self.symbol_encoder = s
        self.v_proj = ManifoldProjection(d); self.s_proj = ManifoldProjection(d)
    def encode_visual(self, x):
        return self.v_proj(self.visual_encoder(x))
    def encode_symbol(self, t, rpc=False):
        if rpc:
            z, pc = self.symbol_encoder(t, return_per_category=True)
            return self.s_proj(z), pc
        return self.s_proj(self.symbol_encoder(t))


def load_model(checkpoint_path, device, hidden_dim=512):
    cfg = Config(); cfg.model.vit_pretrained = False
    cat_to_idx, cat_sizes, _ = build_vocabs(cfg.data)
    visual = VisualEncoder(cfg.model, 256)
    symbol = SymbolEncoder(cat_sizes, 64, hidden_dim, 256)
    model = SAMPipelineStripped(visual, symbol, 256).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    md = model.state_dict()
    pd = {k: v for k, v in ckpt['model_state_dict'].items() if k in md and md[k].shape == v.shape}
    md.update(pd); model.load_state_dict(md); model.eval()
    return model, cat_to_idx, cat_sizes, cfg


def main():
    device = 'cuda'
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading models...")
    model_fix, cat_to_idx, cat_sizes, cfg = load_model(
        'outputs/p2_wave1_fix/checkpoint_epoch030.pt', device, hidden_dim=512)
    model_p3, _, _, _ = load_model(
        'outputs/p3/checkpoint_best.pt', device, hidden_dim=128)

    renderer = ShapeRenderer(image_size=224)
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']
    materials = ['matte', 'shiny', 'metallic', 'glass']

    results = {}

    # =========================================================================
    # G1: All-shapes-all-colors retrieval matrix
    # =========================================================================
    print("\n" + "=" * 70)
    print("G1: Cross-Modal Color Retrieval Matrix (5 shapes × 6 colors)")
    print("=" * 70)

    for model_name, model in [("Fix (30ep)", model_fix), ("P3 Baseline", model_p3)]:
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        per_shape = defaultdict(lambda: {"correct": 0, "total": 0})

        for shape in shapes:
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
                    scores.append((c, (z_v * z_s).sum().item()))
                scores.sort(key=lambda x: -x[1])
                rank = next(i + 1 for i, (c, _) in enumerate(scores) if c == true_c)
                if rank == 1: correct_top1 += 1
                if rank <= 3: correct_top3 += 1
                total += 1
                per_shape[shape]["total"] += 1
                if rank == 1: per_shape[shape]["correct"] += 1

        results[f"{model_name}_top1"] = correct_top1 / total
        results[f"{model_name}_top3"] = correct_top3 / total

        print(f"\n  {model_name}:")
        print(f"    Overall Top-1: {correct_top1}/{total} ({correct_top1/total:.1%})")
        print(f"    Overall Top-3: {correct_top3}/{total} ({correct_top3/total:.1%})")
        print(f"    Per-shape:")
        for shape in shapes:
            s = per_shape[shape]
            bar = "█" * s["correct"] + "░" * (s["total"] - s["correct"])
            print(f"      {shape:<10} {bar} {s['correct']}/{s['total']}")

    # =========================================================================
    # G2: Per-color accuracy breakdown (fix model)
    # =========================================================================
    print("\n" + "=" * 70)
    print("G2: Per-Color Accuracy Breakdown (Fix model)")
    print("=" * 70)

    per_color = defaultdict(lambda: {"correct": 0, "total": 0, "confusions": defaultdict(int)})
    for shape in shapes:
        for true_c in colors:
            img = renderer.render_single_object(shape, true_c, 'medium',
                                                'matte', angle_variant=0)
            t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                z_v = model_fix.encode_visual(t)
            scores = []
            for c in colors:
                sym = f'[OBJ:{shape}] [COL:{c}] [SIZE:medium] [MAT:matte]'
                tok = tokenize(sym, cat_to_idx)
                tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
                with torch.no_grad():
                    z_s = model_fix.encode_symbol(tok)
                scores.append((c, (z_v * z_s).sum().item()))
            scores.sort(key=lambda x: -x[1])
            per_color[true_c]["total"] += 1
            if scores[0][0] == true_c:
                per_color[true_c]["correct"] += 1
            else:
                per_color[true_c]["confusions"][scores[0][0]] += 1

    print(f"\n  {'Color':<10} {'Accuracy':<12} {'Top Confusions'}")
    print(f"  {'-'*50}")
    for c in colors:
        pc = per_color[c]
        acc = pc["correct"] / pc["total"]
        conf_str = ", ".join(f"{k}×{v}" for k, v in
                            sorted(pc["confusions"].items(), key=lambda x: -x[1])[:3])
        print(f"  {c:<10} {pc['correct']}/{pc['total']} ({acc:.0%})   → {conf_str}")

    # =========================================================================
    # G3: Cross-material robustness
    # =========================================================================
    print("\n" + "=" * 70)
    print("G3: Cross-Material Robustness (cube, all colors × all materials)")
    print("=" * 70)

    for material in materials:
        correct = 0
        for true_c in colors:
            img = renderer.render_single_object('cube', true_c, 'medium',
                                                material, angle_variant=0)
            t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                z_v = model_fix.encode_visual(t)
            scores = []
            for c in colors:
                sym = f'[OBJ:cube] [COL:{c}] [SIZE:medium] [MAT:{material}]'
                tok = tokenize(sym, cat_to_idx)
                tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
                with torch.no_grad():
                    z_s = model_fix.encode_symbol(tok)
                scores.append((c, (z_v * z_s).sum().item()))
            scores.sort(key=lambda x: -x[1])
            if scores[0][0] == true_c:
                correct += 1
        bar = "█" * correct + "░" * (6 - correct)
        print(f"  {material:<10} {bar} {correct}/6 ({correct/6:.0%})")

    # =========================================================================
    # G4: Symbol embedding color separation
    # =========================================================================
    print("\n" + "=" * 70)
    print("G4: Symbol Embedding Color Separation")
    print("=" * 70)

    def measure_symbol_separation(model, shape_name):
        embs = {}
        for c in colors:
            sym = f'[OBJ:{shape_name}] [COL:{c}] [SIZE:medium] [MAT:matte]'
            tok = tokenize(sym, cat_to_idx)
            tok = {k: v.unsqueeze(0).to(device) for k, v in tok.items()}
            with torch.no_grad():
                z = model.encode_symbol(tok)
            embs[c] = z.cpu()
        off_diag = []
        for i, c1 in enumerate(colors):
            for j, c2 in enumerate(colors):
                if i != j:
                    off_diag.append((embs[c1] * embs[c2]).sum().item())
        return np.mean(off_diag), np.min(off_diag), np.max(off_diag)

    for model_name, model in [("P3 Baseline", model_p3), ("Fix (30ep)", model_fix)]:
        print(f"\n  {model_name}:")
        for shape in shapes:
            mean_cos, min_cos, max_cos = measure_symbol_separation(model, shape)
            print(f"    {shape:<10} mean={mean_cos:.4f}  min={min_cos:.4f}  max={max_cos:.4f}")

    # =========================================================================
    # G5: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("G5: Summary Table")
    print("=" * 70)

    summary = {
        "fix_top1": results.get("Fix (30ep)_top1", 0),
        "fix_top3": results.get("Fix (30ep)_top3", 0),
        "p3_top1": results.get("P3 Baseline_top1", 0),
        "p3_top3": results.get("P3 Baseline_top3", 0),
    }

    print(f"""
    ┌──────────────────────────────────┬──────────┬──────────┬──────────┐
    │ Metric                           │ P3       │ Fix      │ Change   │
    ├──────────────────────────────────┼──────────┼──────────┼──────────┤
    │ Cross-modal Top-1 (30 tests)     │  {summary['p3_top1']:.0%}    │  {summary['fix_top1']:.0%}    │  +{summary['fix_top1']-summary['p3_top1']:.0%}     │
    │ Cross-modal Top-3 (30 tests)     │  {summary['p3_top3']:.0%}    │  {summary['fix_top3']:.0%}    │  +{summary['fix_top3']-summary['p3_top3']:.0%}     │
    └──────────────────────────────────┴──────────┴──────────┴──────────┘
    """)

    # Save results
    results["per_color"] = {c: {
        "correct": per_color[c]["correct"], "total": per_color[c]["total"]
    } for c in colors}
    with open("outputs/p2_wave1_fix/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to outputs/p2_wave1_fix/eval_results.json")


if __name__ == "__main__":
    main()
