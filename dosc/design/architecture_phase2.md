# SAM Phase 2: From Proof-of-Concept to Robust Foundation

**版本**: v2.0 | **日期**: 2026-06-03 | **状态**: Design Phase

---

## 0. 项目定位

Phase 1 证明了 H1 在理想条件下成立——共享流形使组合系统性和跨模态类比作为几何性质涌现。但暴露了六个结构性缺陷：

1. **颜色盲**：同形状不同颜色的物体在流形上无法有效区分（绿色 bias）
2. **对齐捷径**：L_align 过强导致 L_analogy 退化为恒等映射
3. **评估妥协**：batch-64 检索高估了类比能力
4. **无自然语言**：符号编码器只能处理结构化 token
5. **域差距显著**：真实照片上形状信号尚可，颜色和关系完全失效
6. **未验证扩展性**：5.78M 参数下的结论能否保持到更大规模？

**Phase 2 目标**：修复上述结构性缺陷，将 SAM 从"理想条件下的概念验证"升级为"有一定鲁棒性的地基系统"，为 Phase 3 的大规模扩展做好准备。

**纲领性延续**：
- 8GB VRAM 硬约束不变
- 共享流形核心理念不变——推理即流形动力学
- Phase 2 是补短板，不是推翻重来

---

## 1. 理论基础扩展

### 1.1 颜色特异性对比学习

Phase 1 的 InfoNCE 对比在全局嵌入上做，形状信号主导了颜色信号。Phase 2 引入**属性条件对比**：

$$\mathcal{L}_{\text{color-contrast}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k) / \tau)}$$

其中 $z_j^+$ 是与 $z_i$ **同形状不同颜色**的正样本。这强制模型在固定形状的条件下去区分颜色方向。

**实现**：每个 batch 确保包含同形状多颜色的样本（颜色均衡采样），在流形上计算颜色子空间内的对比损失。

### 1.2 反捷径训练策略

对齐捷径的本质是优化动力学问题：L_align 收敛速度远快于 L_analogy，导致 L_analogy 在 L_align 已收敛的空间中寻找捷径。

Phase 2 采用**阶段性冻结 + 交替优化**：

- **Phase A（恢复期）**：冻结 $E_v$ 和 $E_s$ 的底层参数（前 8 层 ViT + embedding 层），仅训练投影头和上层。降低 L_align 权重至 0.3。
- **Phase B（类比强化期）**：L_analogy 权重提升至 2.0，L_align 降至 0.1。在类比 loss 中加入**属性扰动正则**——随机替换符号中的一个属性值，要求模型检测到不一致。
- **Phase C（联合微调）**：恢复所有 loss 权重，小 LR 联合优化。

### 1.3 自然语言到流形的映射

Phase 1 的符号编码器：
$$E_s: \text{"[OBJ:cube] [COL:red]"} \mapsto \text{Per-cat Embedding} \to \text{Sum Pool} \to \text{MLP} \to \mathcal{M}$$

Phase 2 扩展为双通道符号编码器：

$$E_s^{\text{dual}} = \alpha \cdot E_s^{\text{structured}} + (1-\alpha) \cdot E_s^{\text{language}}$$

其中：
- $E_s^{\text{structured}}$ 保留 Phase 1 的结构化编码器（精度高，用于训练）
- $E_s^{\text{language}}$ 新增的自然语言编码器：frozen sentence-transformer（如 all-MiniLM-L6-v2, 22M 参数）→ 可训练的 Linear 投影到 256-dim 流形

**训练策略**：以结构化编码器为 anchor，用知识蒸馏训练语言编码器——$\mathcal{L}_{\text{distill}} = \|E_s^{\text{structured}}(S) - E_s^{\text{language}}(T)\|^2$，其中 $T$ 是 $S$ 的自然语言表述（如 "a red cube"）。

### 1.4 连续属性的流形平滑性

Phase 1 只验证了离散属性。Phase 2 引入连续属性验证：

**假设**：如果流形上的属性方向是真正的向量场，那么沿该方向连续移动应该产生平滑变化的表征。

**验证协议**：
- 连续颜色：RGB 从 (255,0,0) 到 (0,0,255) 均匀采样 20 个中间色
- 连续位置：物体从 x=-0.5 到 x=+0.5 均匀移动 20 个位置
- 测量：相邻点的 cosine 距离方差、PCA 轨迹的曲率

---

## 2. 架构变更

### 2.1 总览

```text
                    ┌─────────────────────────────────┐
                    │      Shared Manifold M          │
                    │      dim = 256 (unchanged)      │
                    └────┬───────────────────┬────────┘
                         │                   │
              ┌──────────┴───────┐ ┌─────────┴──────────────┐
              │  Visual Encoder  │ │ Dual Symbol Encoder    │
              │  ViT-Tiny/16     │ │                        │
              │  + Color Aug     │ │ ┌────────────────────┐ │
              │  (unchanged)     │ │ │ Structured (Phase1)│ │
              └──────────────────┘ │ │ Per-cat + Pool+MLP │ │
                                   │ └────────────────────┘ │
                                   │ ┌────────────────────┐ │
                                   │ │ Language (NEW)     │ │
                                   │ │ Frozen MiniLM-L6   │ │
                                   │ │ + Trainable Proj   │ │
                                   │ └────────────────────┘ │
                                   └────────────────────────┘
```

### 2.2 颜色增强管线

| 组件 | 方案 | 参数 |
|------|------|------|
| 颜色抖动 | 训练时随机 HSV 偏移 | H:±5°, S:±15%, V:±10% |
| 颜色均衡采样 | 每个 batch 强制 6 种颜色各 ≥1 个样本 | batch_size 从 8 增加到 12 |
| 颜色特异性对比 | 同形状不同颜色的正样本对 | τ_color=0.1 |

### 2.3 自然语言符号编码器

| 项目 | 选择 | 理由 |
|------|------|------|
| Backbone | all-MiniLM-L6-v2 (22M) | 轻量、多语言、384-dim |
| 冻结策略 | 全部冻结 | 不增加可训练参数，仅训练投影 |
| 投影层 | Linear(384→256) + LayerNorm | 映射到共享流形 |
| 蒸馏训练 | MSE with structured encoder | 以结构化编码器为 teacher |
| 参数增量 | +0.1M 可训练 | 几乎不影响显存 |

**自然语言输入格式**：
- 结构化：`[OBJ:cube] [COL:red] [SIZE:medium] [MAT:matte]`（保持不变）
- 自然语言：`"a red medium cube with matte finish"`
- 场景级：`"a blue cube is to the left of an orange cube"`

### 2.4 反捷径训练调度器

```python
class AntiShortcutScheduler:
    """Manages the alternation between alignment and analogy phases."""

    def __init__(self):
        self.phases = [
            # (epochs, L_align_weight, L_rel_weight, L_analogy_weight, L_dis_weight)
            ("recovery",  5,  0.3, 1.0, 0.0, 0.0),   # Fix alignment gently
            ("analogy",  15,  0.1, 0.5, 2.0, 0.1),   # Push analogy hard
            ("joint",    10,  0.5, 1.0, 1.0, 0.1),   # Joint finetune
        ]
```

**关键创新——属性扰动正则**（在 analogy phase 中启用）：

随机选取 analogy 四元组中的符号 $S_A$，以 30% 概率替换其中一个属性值为随机值，形成 $\tilde{S}_A$。计算：

$$\mathcal{L}_{\text{perturb}} = \max(0, \cos(E_v(I_B) - E_v(I_A) + E_s(\tilde{S}_A), E_s(S_B)) - m)$$

其中 $m$ 是 margin。如果扰动了属性后模型仍然预测正确（cos 高），说明它在走对齐捷径而非真正的属性级推理。这个 loss **惩罚捷径行为**。

---

## 3. 数据策略扩展

### 3.1 连续属性数据集

在现有 2D 渲染管线基础上扩展：

| 新增数据 | 生成方式 | 数量 | 目的 |
|----------|---------|------|------|
| 连续颜色 | RGB 空间均匀采样 20 个插值点 × 5 形状 | 100 张 | 测试流形颜色方向平滑性 |
| 连续位置 | x 从 -0.5 到 +0.5 采样 20 个位置 × 3 关系 | 60 张 | 测试关系方向连续可导航性 |
| 连续大小 | scale 从 0.3 到 1.0 采样 15 个 | 15 张 | 测试大小属性线性结构 |

这些数据**仅用于评估，不参与训练**——验证的是流形的零样本连续插值能力。

### 3.2 真实图片微调集

| 类别 | 数量 | 采集要求 |
|------|------|---------|
| 纯色背景单物体 | 30 张 (5 形状 × 6 颜色) | 白/灰背景，自然光，无透视 |
| 自然背景单物体 | 30 张 | 任意背景，日常光照 |
| 双物体场景 | 20 张 | 简单空间关系（左右/上下） |

总计约 80 张真实照片。用途：**fine-tune 验证域适应，非大规模训练**。

标注格式：每张照片配结构化符号（手动标注）。

### 3.3 自然语言配对

对于每个训练样本，自动生成自然语言变体：

```
结构化: [OBJ:cube] [COL:red] [SIZE:medium] [MAT:matte]
    ↓ (模板生成)
NL变体:
  - "a red cube"
  - "a medium-sized red cube with matte finish"
  - "a matte red cube of medium size"
  - "there is a red cube"
```

使用 5-8 个模板覆盖所有属性组合，生成约 1,800 个 NL 变体用于蒸馏训练。

### 3.4 全库检索评估集

Phase 1 的 E3 用的是 batch-64 检索。Phase 2 构建独立的检索评估集：

- **候选库**：预计算所有 360 种单物体组合 + 1000 种场景组合的符号嵌入
- **查询集**：500 个 OOD 类比查询
- **评估**：在完整候选库中检索 Top-1/Top-10/Top-100

这给出的是**真实的全库检索性能**，不依赖于 batch 内的"幸运候选"。

---

## 4. 训练协议 v2

### 4.1 损失函数

沿用 Phase 1 的四项损失，新增两项：

$$L_{\text{P2}} = L_{\text{align}} + \alpha L_{\text{rel}} + \beta L_{\text{analogy}} + \gamma L_{\text{disentangle}} + \delta L_{\text{color-contrast}} + \epsilon L_{\text{perturb}} + \zeta L_{\text{distill}}$$

| 损失 | 符号 | Phase 1 | Phase 2 | 作用阶段 |
|------|------|---------|---------|---------|
| 对齐 | $L_{\text{align}}$ | ✓ | ✓ | All |
| 关系 | $L_{\text{rel}}$ | ✓ | ✓ | Relation+ |
| 类比 | $L_{\text{analogy}}$ | ✓ | ✓ | Analogy+ |
| 解耦 | $L_{\text{disentangle}}$ | ✓ | ✓ | Analogy+ |
| **颜色对比** | $L_{\text{color-contrast}}$ | — | **NEW** | Warmup+ |
| **扰动正则** | $L_{\text{perturb}}$ | — | **NEW** | Analogy only |
| **蒸馏** | $L_{\text{distill}}$ | — | **NEW** | NL training only |

### 4.2 三阶段训练

| 阶段 | Epochs | 数据 | 核心变更 |
|------|--------|------|---------|
| **Recovery** | 1-5 | 合成单物体 + 场景 | 降低 L_align (0.3)，加入 L_color-contrast (0.5)。修复 Phase 1 的颜色盲和对齐捷径。从 Phase 1 best checkpoint 续训。 |
| **Analogy Push** | 6-20 | 合成场景 + 类比 | L_analogy 提升至 2.0，L_align 降至 0.1。激活 L_perturb (0.3)。L_rel 降至 0.5（给类比让路）。 |
| **Joint + NL** | 21-30 | 合成 + NL 变体 + 真实图片 | 恢复均衡权重。加入 L_distill (1.0) 打开 NL 通道。真实图片以低 LR (1e-5) 微调。 |

### 4.3 优化配置（变更部分）

| 项目 | Phase 1 | Phase 2 |
|------|---------|---------|
| Batch size (effective) | 32 | 48 (micro=12, accum=4) |
| Color-balanced sampling | 无 | 是（每 batch 颜色≥1） |
| NL distillation LR | — | 1e-4 (distill only) |
| Real image fine-tune LR | — | 1e-5 |
| 预计训练时间 | ~45 min | ~90 min |
| 预计 VRAM | 183 MB | ~350 MB |

---

## 5. 评估体系 v2

### 5.1 E1: 连续属性插值（新增）

**问题**：流形上的属性方向是否形成平滑的向量场？

**协议**：
1. 沿连续颜色方向（红→蓝 20 步），每一步生成图像 + 符号嵌入
2. 计算相邻步的 cosine 距离和 PCA 主方向方差
3. 沿连续位置方向（左→右 20 步），同上

**成功基准**：相邻步 cosine 距离方差 < 0.01，PCA 主方向在 10° 以内。

### 5.2 E2: 全库检索类比（升级 E3）

**问题**：在完整候选库中，零样本类比的实际检索精度是多少？

**协议**：
1. 预计算所有 360 单物体 + 1000 场景符号嵌入作为候选库
2. 500 个 OOD 类比查询，在完整候选库中检索
3. 测量 Top-1、Top-10、Top-100 准确率

**成功基准**：Top-10 > 50%，证明向量运算的方向在大致正确。

### 5.3 E3: 自然语言对齐（新增）

**问题**：自然语言能否被映射到同一个流形上的正确位置？

**协议**：
1. 用 NL 编码器编码 "a red cube"
2. 用视觉编码器编码红色方块图像
3. 测量跨模态 cosine（NL→视觉 和 结构化→视觉 的差距）

**成功基准**：NL-visual cosine > 0.5（结构化-visual 的 50% 以上）。

### 5.4 E4: 真实图片微调效果（新增）

**问题**：少量真实图片 fine-tune 能否显著提升域适应？

**协议**：
1. Fine-tune 前：在 80 张真实图片上测形状/颜色分类准确率
2. Fine-tune 后：同上
3. 保留 20 张作为 held-out 测试（不参与 fine-tune）

**成功基准**：Fine-tune 后形状准确率 > 80%，颜色准确率 > 40%。

### 5.5 E5-E7: 沿用的 Phase 1 评估

| 实验 | Phase 1 | Phase 2 |
|------|---------|---------|
| E5 组合泛化 (原 E1) | gap 2.4% | gap < 5% + 颜色 gap 独立测量 |
| E6 跨模态同构 (原 E2) | RSA ρ=0.957 | 维持 > 0.9，加入语言 RSA |
| E7 扩展性测试 | 无 | ViT-Small (22M) 相同训练，测量 ρ 和 gap 变化 |

### 5.6 Baseline 对比（新增）

| Baseline | 比较维度 | 预期 |
|----------|---------|------|
| CLIP ViT-B/32 zero-shot | NL→图像检索 | SAM 应在组合性上显著优于 CLIP |
| CLIP + Linear Probe | 同数据 fine-tune | SAM 在 OOD 上应有更低 gap |
| Mini-MLLM (LLaVA-7B 量化版) | 类比推理 | SAM 的结构性优势，LLaVA 的规模优势 |

---

## 6. 代码结构变更

```text
sam/                                  # 主代码包
├── config.py                         # +Phase2Config, +ColorContrastConfig
├── data/
│   ├── renderer.py                   # +连续属性渲染（连续颜色、位置）
│   ├── scene_generator.py            # +连续属性生成、NL 变体生成
│   ├── dataset.py                    # +ColorBalancedSampler
│   └── real_image_dataset.py         # NEW: 真实图片加载器
├── encoders/
│   ├── visual.py                     # (+ color augmentation)
│   ├── symbol.py                     # (unchanged)
│   └── language.py                   # NEW: frozen MiniLM + 可训练投影
├── losses/
│   ├── color_contrastive.py          # NEW: 颜色特异性对比损失
│   └── perturb.py                    # NEW: 属性扰动正则
├── trainer.py                        # +AntiShortcutScheduler, +distill loop
├── eval/
│   ├── continuous_interpolation.py   # NEW: E1 连续属性插值
│   ├── full_corpus_retrieval.py      # NEW: E2 全库检索
│   ├── language_alignment.py         # NEW: E3 自然语言对齐
│   └── real_image_transfer.py        # NEW: E4 真实图片域适应
scripts/
├── p2_train.py                       # NEW: Phase 2 训练入口
├── p2_eval.py                        # NEW: Phase 2 评估入口
├── generate_continuous_data.py       # NEW: 连续属性数据生成
├── generate_nl_variants.py           # NEW: NL 变体生成
├── collect_real_images.py            # NEW: 真实图片标注工具
└── download_language_encoder.py      # NEW: MiniLM 模型下载
data/
├── sam_dataset/                      # Phase 1 数据（保留）
├── continuous/                       # NEW: 连续属性测试数据
├── nl_variants/                      # NEW: NL 变体 JSON
└── real_images/                      # NEW: 真实照片 + 标注
```

---

## 7. 三波推进计划

### Wave 1: 结构性修复（优先级最高）

| 任务 | 预计时间 | 验收标准 |
|------|---------|---------|
| 颜色均衡采样 + L_color-contrast | 2 天 | 合成数据颜色 Top-1 > 50% |
| AntiShortcutScheduler + L_perturb | 2 天 | L_analogy train/val gap < 5x |
| 全库检索评估 | 1 天 | 全库 Top-10 > 50% |
| 重新训练 Phase 2 模型 | 1 天 | 所有 P1 指标不退化 + color 指标达标 |

**Wave 1 完成标志**：颜色盲和对齐捷径修复，全库检索替换 batch 检索。

### Wave 2: 能力拓展

| 任务 | 预计时间 | 验收标准 |
|------|---------|---------|
| 自然语言编码器 + 蒸馏训练 | 3 天 | NL-visual cosine > 0.5 |
| 真实图片采集 + fine-tune | 2 天 | Fine-tune 后形状 > 80%, 颜色 > 40% |
| 连续属性数据 + 评估 | 1 天 | 相邻插值 cosine 方差 < 0.01 |
| 连接语言 RSA (NL→Visual vs Struct→Visual) | 1 天 | NL 的 RSA ρ > 0.5 |

**Wave 2 完成标志**：NL 通道打通，真实图片域适应验证。

### Wave 3: 巩固闭环

| 任务 | 预计时间 | 验收标准 |
|------|---------|---------|
| ViT-Small 扩展性测试 | 2 天 | 参数 4x 后 RSA ρ 和 IID-OOD gap 保持 |
| 轻量文本解码器 | 3 天 | 流形→自然语言生成可读性验证 |
| CLIP / Mini-MLLM baseline 对比 | 2 天 | 量化 SAM vs Transformer 的 OOD gap 差异 |
| Phase 2 论文完整版 | 3 天 | 包含所有 Phase 1+2 数据，提交 arXiv |

---

## 8. 风险与应对

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| L_color-contrast 仍无法修复颜色盲 | 中 | 高 | 尝试显式的颜色分类头 (auxiliary task)；降低流形维度以增大颜色信号密度 |
| L_perturb 与 L_analogy 冲突 | 中 | 中 | 如果冲突明显，将 L_perturb 改为评估期惩罚，不做训练期 loss |
| MiniLM 冻结后投影层容量不足 | 低 | 中 | 解冻 MiniLM 最后 2 层，或换用更小但可训练的 Transformer |
| 真实图片 fine-tune 过拟合 | 中 | 中 | 强正则化 (dropout=0.5, weight_decay=0.1)，early stopping |
| 全库检索导致评估时间过长 | 低 | 低 | 预计算所有候选嵌入 (30s 完成)，检索只需一次矩阵乘法 |
| ViT-Small 超出 8GB VRAM | 中 | 高 | 使用 gradient checkpointing + 更激进的冻结策略 |

---

## 9. 里程碑总览

```
Phase 2: 共 6 周

W1-W2:  Wave 1 — 结构性修复
W3-W4:  Wave 2 — 能力拓展
W5:     Wave 3 — 巩固闭环
W6:     论文整合 + 代码发布
```

| 里程碑 | 周次 | 交付物 |
|--------|------|--------|
| M1: 颜色盲修复 | W2 | 合成数据颜色准确率 > 50% |
| M2: 捷径修复 | W2 | L_analogy gap < 5x |
| M3: NL 通道打通 | W4 | NL→视觉 cosine > 0.5 |
| M4: 真实图片验证 | W4 | 域适应形状 > 80% |
| M5: 扩展性验证 | W5 | ViT-Small 保持几何特性 |
| M6: Phase 2 论文 | W6 | 完整论文 + 代码开源 |

---

## 10. 与 Phase 1 的关系

Phase 2 不是对 Phase 1 的否定，而是对 Phase 1 暴露问题的系统性回应：

| Phase 1 暴露的问题 | Phase 2 的回应 |
|-------------------|---------------|
| 颜色盲 | L_color-contrast + 颜色均衡采样 |
| 对齐捷径 | AntiShortcutScheduler + L_perturb |
| batch 检索高估 | 全库检索评估 |
| 无自然语言 | 双通道符号编码器 (struct + NL) |
| 域差距 | 真实图片 fine-tune + 评估 |
| 无扩展验证 | ViT-Small 对照实验 |
| 无 baseline 对比 | CLIP + Mini-MLLM 对比 |

Phase 1 证明了机制存在。Phase 2 证明机制健壮。Phase 3 将其推向大规模。
