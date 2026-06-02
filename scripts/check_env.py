"""Quick environment check for SAM."""
import torch
import timm
import torchvision
import numpy as np
from PIL import Image

print(f"PyTorch:    {torch.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM:       {vram:.1f} GB")
print(f"timm:       {timm.__version__}")
print(f"torchvision:{torchvision.__version__}")
print(f"numpy:      {np.__version__}")
print(f"Pillow:     {Image.__version__}")
print(f"matplotlib: ", end="")
import matplotlib; print(matplotlib.__version__)
