"""Check dataset generation progress."""
from pathlib import Path

d = Path('E:/PythonProject/mm-jepa/data/sam_dataset')
for sub in ['single_objects', 'images/train', 'images/val',
             'images/test_iid', 'images/test_ood']:
    p = d / sub
    if p.exists():
        n = len(list(p.glob('*.png')))
        print(f'  {sub}: {n} images')
    else:
        print(f'  {sub}: not created yet')
