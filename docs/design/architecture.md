# SAM: Shared Analogical Manifold — Architecture & Programmatic Design

**版本**: v1.1 | **日期**: 2026-06-02 | **状态**: P0 Complete, P1 Pending

---

## 0. 项目身份

**SAM (Shared Analogical Manifold)** 是一个多模态原生推理的概念验证系统。它探索一种不同于 Transformer 自回归范式的核心机制：**让视觉和符号在同一个连续流形上共存，推理即流形上的动力学演化**。

**核心假设 (H1):** 如果视觉表征和符号表征被训练为共享同一个解耦的连续流形，那么跨模态类比推理和组合系统性泛化将作为流形的几何性质自然涌现——无需额外的推理模块，无需庞大的参数量。

**纲领性原则:**
- 符号不是视觉的翻译目标，而是流形上同等的一等公民
- 推理发生在流形上，而非 token 序列中
- 组合性来自流形的线性结构，而非注意力模式的统计插值
- 8GB VRAM 硬约束下验证核心机制，不为规模牺牲设计纯度

---

## 1. 理论基础

### 1.1 共享流形假设

令 $\mathcal{M} \subset \mathbb{R}^d$ 为共享表示流形。

定义两个编码器:
- $E_v: \mathcal{I} \to \mathcal{M}$ — 视觉编码器，将图像映射到流形
- $E_s: \mathcal{S} \to \mathcal{M}$ — 符号编码器，将结构化符号映射到流形

**关键约束:** $E_v$ 和 $E_s$ 的输出空间是**同一个** $\mathcal{M}$，不是两个对齐的独立空间。

### 1.2 组合系统性

流形 $\mathcal{M}$ 被假设为一个近似的**直和结构**:

$$\mathcal{M} \approx \mathcal{A}_1 \oplus \mathcal{A}_2 \oplus ... \oplus \mathcal{A}_k \oplus \mathcal{O} \oplus \mathcal{R}$$

其中:
- $\mathcal{A}_i$ — 第 i 个属性的子空间（颜色、大小、材质...）
- $\mathcal{O}$ — 物体身份子空间
- $\mathcal{R}$ — 关系子空间

**组合系统性 = 任意基向量的线性组合仍然落在 $\mathcal{M}$ 上一个合法的点。**

如果模型见过 `v(red_cube)` 和 `v(blue_sphere)`，它应该自动能表示 `v(blue_cube)`——因为 "blue" 方向 + "cube" 方向 = blue_cube 在流形上的位置。

### 1.3 跨模态类比 = 流形上的向量运算

类比映射在流形上表现为位移向量的对齐:

$$\vec{d}_v = E_v(I_a) - E_v(I_b)$$
$$\vec{d}_s = E_s(S_a) - E_s(S_b)$$
$$\text{cosine}(\vec{d}_v, \vec{d}_s) \to 1$$

以及跨模态操作:
$$E_v(I_{\text{red\_cube}}) - E_s(\text{"red"}) + E_s(\text{"blue"}) \approx E_v(I_{\text{blue\_cube}})$$

---

## 2. 架构设计

### 2.1 总览

```text
                   ┌──────────────────────────────────┐
                   │         Shared Manifold M        │
                   │         dim = 256                │
                   │                                  │
                   │  ●v(cube)  ●v(sphere)           │
                   │  ★t(cube)  ★t(sphere)           │
                   │                                  │
                   │  属性方向: →color  →size  →mat   │
                   │  物体方向: →obj_1  →obj_2        │
                   │  关系方向: →left  →above  →in    │
                   └───┬──────────────────────┬───────┘
                       │                      │
              ┌────────┴────────┐    ┌────────┴────────┐
              │  Visual Encoder │    │ Symbol Encoder  │
              │  ViT-Tiny/16    │    │  AttrEmbed+MLP  │
              │  ~5.5M params   │    │  ~1.5M params   │
              └────────┬────────┘    └────────┬────────┘
                       │                      │
              ┌────────┴────────┐    ┌────────┴────────┐
              │   Input Image   │    │ Structured Sym  │
              │   224 × 224     │    │ [OBJ:cube]      │
              │                 │    │ [COL:red] ...   │
              └─────────────────┘    └─────────────────┘
```

### 2.2 视觉编码器 $E_v$

| 项目 | 选择 | 理由 |
|------|------|------|
| Backbone | ViT-Tiny/16 (5.5M) | 8GB VRAM 约束下的最优选择 |
| 预训练 | ImageNet-1k 权重 (ModelScope) | 避免从零学底层特征；通过 `scripts/download_vit_weights.py` 从 ModelScope 下载，缓存于 `~/.cache/sam/pretrained/`。设置 `SAM_PRETRAINED_VIT` 环境变量指向下载目录，若未设置则退回到 HuggingFace Hub |
| Patch size | 16×16 | 224/16 = 14×14 = 196 patches |
| 输出方式 | CLS token → Linear(192→256) | 全局表示，维度匹配流形 |
| 冻结策略 | 前 8 层冻结，后 4 层可训练 | 保留底层特征，允许高层适应流形 |
| 精度 | bfloat16 | 节省显存 |

### 2.3 符号编码器 $E_s$

| 项目 | 选择 | 理由 |
|------|------|------|
| 输入格式 | 结构化 token 序列 | v1 不做自然语言，保证编码器极简 |
| Token 定义 | 见 §3.1 | 每个属性值有独立 token |
| Embedding | Per-category embedding → sum pooling | 组合性由加法体现 |
| MLP head | Linear(64→128→256) + LayerNorm | 轻量投影到流形 |
| 参数总量 | ~1.5M | 极小，几乎不影响显存 |

**符号输入例:**
```
[OBJ:cube] [COL:red] [SIZE:medium] [MAT:matte]
[REL:left_of]
[OBJ:sphere] [COL:blue] [SIZE:small] [MAT:shiny]
```

每个括号是一个 token，属性类别和属性值共享嵌入表。

### 2.4 流形投影层

两个编码器共享一个**流形正则化层**:

```python
class ManifoldProjection(nn.Module):
    def __init__(self, dim=256):
        self.proj = nn.utils.parametrizations.spectral_norm(
            nn.Linear(dim, dim)
        )
        self.ln = nn.LayerNorm(dim)
    
    def forward(self, x):
        # 将编码器输出正则化到单位超球面上
        return F.normalize(self.ln(self.proj(x)), dim=-1)
```

投影到单位超球面 $S^{d-1}$ 有几个好处:
- 训练更稳定（不会发散）
- 距离自然变成角度距离
- 流形结构更清晰（球面上的线性结构即大圆）

---

## 3. 数据策略

### 3.1 属性定义

| 类别 | Token 值 | 数量 |
|------|---------|------|
| 物体 (OBJ) | cube, sphere, cylinder, cone, pyramid | 5 |
| 颜色 (COL) | red, blue, green, yellow, purple, orange | 6 |
| 大小 (SIZE) | small, medium, large | 3 |
| 材质 (MAT) | matte, shiny, metallic, glass | 4 |
| 关系 (REL) | left_of, right_of, above, below, near, far | 6 |

**组合空间大小:** $5 \times 6 \times 3 \times 4 = 360$ 种单物体配置
**双物体 + 关系:** $360 \times 360 \times 6 \approx 777,600$ 个可能场景

### 3.2 组合泛化切分

**关键设计:** 按属性值组合切分，而非按场景实例切分。

| 集合 | 内容 | 占比 |
|------|------|------|
| 训练 | 物体×(颜色子集1 + 大小子集1 + 材质子集1) | ~25% |
| 验证 | 物体×(颜色子集2) — 新颜色组合 | ~15% |
| 测试-IID | 训练分布中的新场景 | ~20% |
| 测试-OOD | 全新属性组合（如训练时未出现的颜色-材质配对） | ~40% |

**检验:** 如果模型在 Test-OOD 上的类比准确率远高于基于统计匹配的 baseline，说明它学到了组合结构而非记忆模式。

### 3.3 数据生成管线

**方案:** 程序化 2D 渲染（pymunk + pycairo / matplotlib）

```
Phase 1: 单物体渲染
  - 渲染每种 (OBJ, COL, SIZE, MAT) 组合的独立图像
  - 约 360 张，每张 224×224
  - 不同 camera angles (2-3 个) → ~1000 张单物体图

Phase 2: 双物体场景渲染  
  - 从单物体库中采样一对 (obj_a, obj_b)
  - 按 REL token 排布空间位置
  - 渲染双物体场景 → 约 10,000-20,000 张场景图
  - 每张配有结构化的符号描述

Phase 3: 类比样本生成
  - 对于每个场景，构造类比四元组:
    (img_A, sym_A, img_B) → sym_B
    (img_A, sym_A, sym_B) → img_B (可选，需要解码器)
  - v1 只做前向: img → sym
```

**为什么用 2D 而非 3D:**
- 程序化 2D 渲染秒级完成，Blender 3D 渲染慢且调参复杂
- 2D 几何形状足以证明组合系统和类比映射
- 8GB 显存下 2D 图像质量完全够
- 后续可平滑升级到 Blender 3D

### 3.4 数据增强

| 增强 | 参数 | 目的 |
|------|------|------|
| 位置抖动 | ±5% 平移 | 防止位置过拟合 |
| 颜色抖动 | ±10% HSV | 光照鲁棒性 |
| 尺度抖动 | ±10% | 尺度不变性 |
| 背景噪声 | 随机纹理 | 防止背景捷径 |

---

## 4. 训练协议

### 4.1 损失函数

总损失由四个项组成:

$$L = L_{\text{align}} + \alpha L_{\text{rel}} + \beta L_{\text{analogy}} + \gamma L_{\text{disentangle}}$$

#### L_align — 跨模态对齐

对于配对样本 $(I, S)$，要求视觉表征和符号表征在流形上靠近:

$$L_{\text{align}} = 1 - \text{cosine}(E_v(I), E_s(S))$$

同时也做 batch 内的对比（InfoNCE），让非配对样本互相推开。

#### L_rel — 关系一致性

对于两个场景对 $(I_a, I_b)$ 和 $(S_a, S_b)$:

$$L_{\text{rel}} = \|(E_v(I_a) - E_v(I_b)) - (E_s(S_a) - E_s(S_b))\|_2^2$$

这保证了视觉空间中的关系和符号空间中的关系是**同一个方向**。

#### L_analogy — 类比完成

给定 $(I_a, S_a, I_b)$，预测 $S_b$ 应该满足:

$$E_s(S_b^{\text{pred}}) = E_v(I_b) - E_v(I_a) + E_s(S_a)$$
$$L_{\text{analogy}} = 1 - \text{cosine}(E_s(S_b^{\text{true}}), S_b^{\text{pred}})$$

这是端到端的类比推理训练。

#### L_disentangle — 解耦正则

鼓励不同属性类别的方向在流形上正交:

$$L_{\text{disentangle}} = \sum_{i \neq j} |\langle \vec{a}_i, \vec{a}_j \rangle|^2$$

其中 $\vec{a}_i$ 是属性类别 i 在流形上的平均方向。这推动了 §1.2 中的直和结构。

#### 超参数

| 参数 | 值 | 理由 |
|------|-----|------|
| α (L_rel 权重) | 1.0 | 关系一致性是核心诉求 |
| β (L_analogy 权重) | 1.0 | 类比能力是主要输出 |
| γ (L_disentangle 权重) | 0.1 | 辅助项，不宜过大 |

### 4.2 课程学习

| 阶段 | 轮次 | 内容 | 激活的 Loss |
|------|------|------|------------|
| Warmup | 1-5 epoch | 单物体对齐 | L_align only |
| Relation | 6-20 epoch | 引入双物体场景和关系 | L_align + L_rel |
| Analogy | 21-40 epoch | 引入类比完成任务 | 全部四项 |
| Fine-tune | 41-60 epoch | 降低 lr，专注难例 | 全部四项 |

### 4.3 优化配置

| 项目 | 值 |
|------|-----|
| Optimizer | AdamW (β1=0.9, β2=0.999) |
| LR schedule | Cosine annealing, warmup 5% steps |
| Peak LR | 1e-4 (encoder), 1e-3 (projection heads) |
| Weight decay | 0.05 |
| Batch size | 32 (micro=8, accum=4) |
| Epochs | 60 |
| Precision | bfloat16 |
| Estimated VRAM | ~4.5 GB peak |
| Estimated time | ~4-6 hours on RTX 4060 |

---

## 5. 评估体系

### 5.1 E1: 组合泛化 (Compositional Generalization)

**问题:** 模型能否正确表征训练中从未见过的属性组合？

**协议:**
1. 在 Test-OOD 集合上，给定新组合的 $(I, S)$ 对
2. 计算 $E_v(I)$ 和 $E_s(S)$ 的 cosine similarity
3. 与纯视觉 baseline 和随机 baseline 对比

**指标:** Top-1 匹配准确率、Recall@10
**成功基准:** OOD 准确率 > 70%，且 IID-OOD gap < 15%

### 5.2 E2: 跨模态关系同构 (Cross-Modal Isomorphism)

**问题:** 视觉关系向量和符号关系向量是否指向同一方向？

**协议:**
1. 对所有测试场景对，计算 $\vec{d}_v$ 和 $\vec{d}_s$
2. 计算 cosine similarity 分布
3. 用 RSA (Representational Similarity Analysis) 比较视觉 RDM 和符号 RDM

**指标:** 平均 cosine similarity、RSA 相关系数
**成功基准:** 平均 cosine > 0.5，RSA ρ > 0.3

### 5.3 E3: 零样本跨模态类比 (Zero-Shot Cross-Modal Analogy)

**问题:** 能否用流形上的向量运算完成未见过的类比？

**协议:**
1. 给出 $(I_a, S_a, I_b)$，预测 $S_b$
2. 计算: $\hat{z}_s = E_v(I_b) - E_v(I_a) + E_s(S_a)$
3. 在符号 token 空间中找最近邻: $\arg\max_S \text{cosine}(\hat{z}_s, E_s(S))$
4. 检查预测的 $S$ 是否与 ground truth $S_b$ 匹配

**指标:** Top-1 / Top-3 准确率
**成功基准:** Top-1 > 40%, Top-3 > 65%

### 5.4 定性分析

- **流形可视化:** UMAP/t-SNE 投影，按属性着色，检查解耦程度
- **属性方向验证:** 计算颜色变化向量在不同物体上的一致性
- **失败案例分析:** 哪些属性组合泛化失败？是否有系统性模式？

### 5.5 Baseline 对比

| Baseline | 说明 | 预期劣势 |
|----------|------|---------|
| CLIP ViT-B/32 (frozen) | 用 CLIP 的 embedding 做同样的类比任务 | 缺少关系一致性训练 |
| 纯视觉 JEPA | 只用视觉自监督，无符号编码器 | 无法做跨模态类比 |
| 随机初始化 ViT+MLP | 同样架构但无流形约束 | 无解耦，无关系对齐 |

---

## 6. 推向 LLM 的扩展接口

PoC 阶段不需要实现这些，但架构必须预留接口。

### 6.1 文本解码器接口

```python
class TextDecoder(nn.Module):
    """从流形点解码为自然语言 token 序列"""
    def __init__(self, manifold_dim=256, vocab_size=32000):
        self.manifold_to_latent = nn.Linear(manifold_dim, hidden_dim)
        self.transformer_decoder = ...  # 轻量 transformer decoder
    
    def forward(self, z_manifold, prompt_tokens=None):
        # z_manifold 是流形上的当前状态
        # 条件生成文本序列
        ...
```

### 6.2 纯文本推理接口

```python
class TextualReasoningInterface:
    """纯文本推理 = 文本 → 流形 → 演化 → 文本"""
    def reason(self, text_input: str) -> str:
        z = self.symbol_encoder.encode(text_input)
        z = self.manifold.evolve(z, steps=N)  # 流形上的动力学
        return self.text_decoder.decode(z)
```

### 6.3 约束满足接口

```python
class ConstrainedManifold:
    """在流形上施加逻辑/数学约束"""
    def project_to_constraint_surface(self, z, constraints):
        # 将 z 投影到满足所有约束的子流形上
        ...
```

这些接口的设计原则：**流形是唯一的状态持有者，所有模态都通过投影进出流形，推理是流形上的动力学。**

---

## 7. 技术风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 流形解耦不充分 | 中 | 高 | L_disentangle 调优；考虑 β-VAE 风格的信息瓶颈 |
| 符号编码器表达能力不足 | 低 | 中 | v1 用结构化 token，维数低，MLP 足够 |
| 类比向量退化到零 | 中 | 中 | L_rel 同时做正样本拉近和负样本推开 |
| 2D 数据过于简单，审稿人挑刺 | 中 | 低 | 设计足够多样的组合空间；保留 Blender 升级路径 |
| 训练不稳定 | 低 | 低 | 单位球面投影 + LayerNorm + warmup |
| 模型只学了物体匹配，没学关系 | 中 | 高 | L_rel 必需；通过关系-only 的探针任务验证 |

---

## 8. 里程碑与交付

| 阶段 | 周期 | 核心任务 | 验收标准 |
|------|------|---------|---------|
| **P0: 基建** | W1 | 搭建代码框架；实现数据生成管线 | 能生成 20k 场景图 + 符号标注；单次 forward <4GB |
| **P1: 对齐** | W2-W3 | 实现 E_v + E_s + L_align；单物体训练 | 单物体对齐准确率 > 90% |
| **P2: 关系** | W4-W5 | 双物体场景；L_rel 训练；关系一致性验证 | E2 指标开始有正信号 (cos > 0.2) |
| **P3: 类比** | W6-W7 | L_analogy 训练；全 loss 联合优化 | E1 组合泛化 > 60%, E3 类比 > 30% |
| **P4: 评估** | W8-W9 | 完整评估三个实验；baseline 对比；可视化 | 产出完整指标报告 |
| **P5: 写作** | W10-W12 | 失败分析；理论升华；撰写研究报告 | 可提交的论文初稿 + 开源代码 |

---

## 9. 代码结构

```text
mm-jepa/
├── docs/
│   ├── plan/
│   │   └── initial_plan.md          # 原始计划 (archived)
│   └── design/
│       └── architecture.md          # 本文档
├── sam/                             # 主代码包
│   ├── __init__.py
│   ├── config.py                    # 全局配置 (dataclass)
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── visual.py               # ViT-Tiny 视觉编码器
│   │   └── symbol.py               # 结构化符号编码器
│   ├── manifold/
│   │   ├── __init__.py
│   │   └── projection.py           # 流形投影 + 正则化
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── align.py                # L_align
│   │   ├── relational.py           # L_rel
│   │   ├── analogy.py              # L_analogy
│   │   └── disentangle.py          # L_disentangle
│   ├── data/
│   │   ├── __init__.py
│   │   ├── renderer.py             # 2D 程序化渲染
│   │   ├── scene_generator.py      # 场景 + 符号配对生成
│   │   └── dataset.py              # PyTorch Dataset
│   ├── trainer.py                   # 训练循环 + 课程学习
│   └── eval/
│       ├── __init__.py
│       ├── compositionality.py     # E1
│       ├── isomorphism.py          # E2
│       └── analogy.py              # E3
├── scripts/
│   ├── generate_data.py            # 数据生成入口
│   ├── train.py                    # 训练入口
│   └── eval.py                     # 评估入口
├── notebooks/
│   └── analysis.ipynb              # 可视化分析
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 10. 结语

SAM 试图回答一个根本性问题：**如果多模态融合不是"翻译"而是"共舞"，会发生什么？**

我们不追求在现有 benchmark 上刷 SOTA。我们追求的是证明存在一套不同的机制——一套让视觉和符号在同一个数学空间中平权协作的机制——它天然地解决了一些 transformer 靠规模也无法根本解决的问题。

PoC 成功后，扩展路径是清晰的：(1) 加文本解码器 → 能对话，(2) 加约束满足 → 能做数学/代码，(3) 扩大规模 → 新范式的大模型。但第一步是证明这个引擎能转。
