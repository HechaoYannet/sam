from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # Object attributes
    objects: List[str] = field(default_factory=lambda: [
        "cube", "sphere", "cylinder", "cone", "pyramid"
    ])
    colors: List[str] = field(default_factory=lambda: [
        "red", "blue", "green", "yellow", "purple", "orange"
    ])
    sizes: List[str] = field(default_factory=lambda: [
        "small", "medium", "large"
    ])
    materials: List[str] = field(default_factory=lambda: [
        "matte", "shiny", "metallic", "glass"
    ])
    relations: List[str] = field(default_factory=lambda: [
        "left_of", "right_of", "above", "below", "near", "far"
    ])

    # Rendering
    image_size: int = 224

    # Split ratios by combination (not instance)
    train_ratio: float = 0.25
    val_ratio: float = 0.15
    test_iid_ratio: float = 0.20
    test_ood_ratio: float = 0.40

    # Generation
    total_scenes: int = 20000
    single_object_variants: int = 3  # camera angles per object

    # v8: Continuous color
    use_continuous_color: bool = False  # enable RGB continuous color
    color_hue_samples: int = 36  # total hues on the wheel
    color_anchor_hues: int = 6   # discrete anchor hues for warmup

    # v8: Structured analogy
    structured_analogy: bool = False  # single-attribute-change analogies
    n_analogy_variants_per_scene: int = 5  # variants per base scene


@dataclass
class ModelConfig:
    # Shared manifold
    manifold_dim: int = 256

    # Visual encoder
    vit_model: str = "vit_tiny_patch16_224"
    vit_pretrained: bool = True
    vit_freeze_layers: int = 8  # freeze first 8 of 12 layers
    image_size: int = 224

    # Symbol encoder
    symbol_embed_dim: int = 64
    symbol_hidden_dim: int = 128
    symbol_use_positional: bool = False  # v1: no positional encoding
    # Token vocabulary sizes (derived from data attributes)
    # These are computed at init from DataConfig


@dataclass
class LossConfig:
    alpha_rel: float = 1.0       # relational consistency weight
    beta_analogy: float = 1.0    # analogy completion weight
    gamma_disentangle: float = 0.1  # disentanglement weight
    alignment_temperature: float = 0.07  # for InfoNCE
    # VICReg + Decoder (Wave 3: scalable regularization)
    use_vicreg: bool = False       # enable VICReg variance/covariance
    vicreg_var_weight: float = 0.5  # variance loss weight
    vicreg_cov_weight: float = 0.5  # covariance loss weight
    use_decoder: bool = False      # enable decoder bottleneck
    decoder_weight: float = 0.5    # decoder cross-entropy weight
    decoder_hidden: int = 128      # decoder hidden dimension

    # v8: Continuous color decoder
    color_rgb_weight: float = 0.1  # MSE weight for RGB regression head


@dataclass
class TrainConfig:
    # Optimization
    lr: float = 1e-4
    lr_projection: float = 1e-3  # higher LR for projection heads
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)

    # Batch
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch = 32

    # Schedule
    epochs: int = 60
    warmup_epochs: int = 3   # 5% of total
    lr_schedule: str = "cosine"

    # Curriculum phases (epoch boundaries)
    phase_warmup_end: int = 5      # L_align only
    phase_relation_end: int = 20   # + L_rel
    phase_analogy_end: int = 40    # + L_analogy + L_disentangle
    phase_finetune_end: int = 60   # all losses, lower lr

    # Precision
    use_amp: bool = True   # bfloat16
    num_workers: int = 0  # 0 = main process only, avoids fork memory overhead
    pin_memory: bool = False  # disabled for num_workers=0
    prefetch_factor: int = 2

    # Logging
    log_interval: int = 50  # steps between logs


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
    device: str = "cuda"
