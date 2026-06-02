"""Check P1 readiness — verify single-object data is available."""
import json
from pathlib import Path
from sam.config import Config
from sam.data.dataset import build_vocabs

cfg = Config()
data_dir = Path('data/test_mini')

with open(data_dir / 'single_objects_meta.json') as f:
    single_meta = json.load(f)

print(f'Single objects: {len(single_meta)}')

obj_types = set()
colors = set()
sizes = set()
materials = set()
for item in single_meta:
    obj_types.add(item['obj_type'])
    colors.add(item['color'])
    sizes.add(item['size'])
    materials.add(item['material'])

print(f'Objects:   {sorted(obj_types)} ({len(obj_types)})')
print(f'Colors:    {sorted(colors)} ({len(colors)})')
print(f'Sizes:     {sorted(sizes)} ({len(sizes)})')
print(f'Materials: {sorted(materials)} ({len(materials)})')
print(f'Total combos: {len(obj_types)}x{len(colors)}x{len(sizes)}x{len(materials)} = {len(obj_types)*len(colors)*len(sizes)*len(materials)}')
print(f'Actual samples: {len(single_meta)} (with 3 variants = {len(single_meta)//3} unique combos)')
print()
print('P1 ready!')
