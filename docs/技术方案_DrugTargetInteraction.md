# BioInteract：可解释的药物-靶点交互预测技术方案

> 项目代号：BioInteract  
> 硬件：单卡 RTX 4080 (16GB VRAM)  
> 目标：发表 SCI 二区/一区  
> 日期：2026-02-10

---

## 一、研究问题——从生物学出发

我们要解决的不是"如何在 benchmark 上刷到更高的 AUROC"，而是一个真正的生物学问题：

**当一个药物分子与一个蛋白质靶点发生结合时，到底是药物的哪个亚结构（药效团）在与蛋白质的哪些关键残基发生相互作用？我们能否让模型"看见"并"解释"这种分子层面的识别机制？**

传统的 DTI 预测模型把药物和蛋白质各自编码成一个向量，然后做点积或拼接——这相当于把一场精密的分子对话压缩成了两个数字。我们丢掉了最有价值的信息：**交互的空间定位和化学语义**。

### 1.1 生物学背景

药物-靶点结合本质上是一种分子识别过程，遵循"锁钥模型"和"诱导契合"原理：

- **结合口袋（Binding Pocket）**：蛋白质表面的一个凹陷区域，通常由 10-30 个关键残基构成
- **药效团（Pharmacophore）**：药物分子上负责结合活性的功能基团组合（氢键供/受体、疏水基团、芳香环等）
- **非共价相互作用**：氢键、疏水作用、π-π 堆积、盐桥、范德华力——这些才是结合的"语言"

我们的模型应该能够回答：预测出的相互作用注意力热图，是否与已知的结合口袋残基一致？关注到的药物亚结构，是否确实是药效团基团？

### 1.2 核心创新定位

| 创新维度 | 具体做法 | 生物学意义 |
|---------|---------|-----------|
| 残基级交互注意力 | Cross-Attention 生成药物原子-蛋白残基交互矩阵 | 定位结合口袋关键残基 |
| 药效团子图识别 | GNN 注意力权重 + Grad-CAM 映射回分子子结构 | 发现活性药效团模式 |
| 蛋白质功能域感知 | 融入 InterPro/Pfam 域注释作为辅助特征 | 理解靶点家族层面的结合偏好 |
| 可解释性验证闭环 | 注意力热图 vs PDB 晶体结构已知结合位点 | 证明模型确实学到了生物学知识 |

---

## 二、模型架构：BioInteract

### 2.1 总体设计

```
                    ┌─────────────────────────────────┐
                    │        BioInteract Model         │
                    └─────────────────────────────────┘

  ┌──────────────┐                                    ┌──────────────────┐
  │  Drug Branch │                                    │  Target Branch   │
  │              │                                    │                  │
  │ SMILES → Mol │                                    │ FASTA → ESM-2   │
  │   Graph      │                                    │  (offline)       │
  │     ↓        │                                    │     ↓            │
  │ GINE(3层)    │                                    │ Residue Embed.   │
  │ + Edge Attr  │                                    │ + Domain Feat.   │
  │     ↓        │                                    │     ↓            │
  │ Atom Repr.   │        ┌──────────────────┐        │ Residue Repr.    │
  │ a∈R^(n×256)  │───────→│  Cross-Attention │←───────│ r∈R^(m×256)     │
  └──────────────┘        │  (Multi-Head)    │        └──────────────────┘
                          │       ↓          │
                          │ Interaction Map  │
                          │ M ∈ R^(n×m)      │
                          │       ↓          │
                          │ Gated Pooling    │
                          │       ↓          │
                          │ Prediction Head  │
                          │ (MLP → σ)        │
                          └──────────────────┘
                                  ↓
                        Binding Probability / 
                        Binding Affinity (pKd)
```

### 2.2 药物分支：分子图 + 化学先验

不是随便建个图就完事。分子图的节点和边特征要融入**化学领域知识**：

**节点（原子）特征：**
- 原子类型（C, N, O, S, F, Cl, ...） — one-hot
- 杂化类型（sp, sp2, sp3）
- 形式电荷
- 芳香性标志
- 是否在环上
- 氢键供/受体能力 ← **药效团相关**
- 疏水性贡献（Crippen logP 贡献） ← **药效团相关**

**边（化学键）特征：**
- 键类型（单/双/三/芳香）
- 是否共轭
- 是否在环中
- 键立体化学（E/Z）

**GNN 选型：GINE (Graph Isomorphism Network with Edge features)**
- 比 GCN 表达力强（能区分 Weisfeiler-Lehman 不可区分的图）
- 支持边特征，能感知化学键类型
- 3 层，每层 256 维，残差连接
- 最后不做全局 readout，保留每个原子的局部表征 → 送入交互模块

### 2.3 靶点分支：ESM-2 + 功能域注释

**ESM-2 特征提取（离线）：**
- 使用 `esm2_t33_650M_UR50D`，输出每个残基的 1280 维表征
- 通过一个可学习的线性投影降到 256 维
- 离线预计算，存为 `.pt` 文件

**功能域辅助特征：**
- 从 InterPro/Pfam 获取蛋白质的域注释
- 每个残基标注其所属功能域类型（如 Kinase domain, SH2 domain, 无域等）
- 编码为可学习 embedding，与 ESM-2 特征拼接
- **生物学意义**：不同功能域有不同的结合偏好，比如激酶域倾向与 ATP 竞争性抑制剂结合

### 2.4 交互模块：交叉注意力 + 可解释性设计

这是整个模型的灵魂。设计原则是：**注意力权重必须有生物学可解释性**。

```
Multi-Head Cross-Attention:

Q = W_q · AtomRepr       (药物原子作为 query)
K = W_k · ResidueRepr    (蛋白残基作为 key)
V = W_v · ResidueRepr

Attention(Q, K, V) = softmax(QK^T / √d_k) · V

交互矩阵 M = mean_over_heads(softmax(QK^T / √d_k))
→ M[i,j] 表示第 i 个药物原子对第 j 个蛋白残基的注意力强度
→ 即模型认为的"交互强度"
```

**Gated Pooling（门控池化）：**
- 不用简单的 mean/max pooling
- 学习一个 gate 向量，让模型自己决定哪些交互 pair 对最终预测最重要
- 这个 gate 本身也是可解释的信号

### 2.5 预测头

- 二分类任务：MLP(256 → 128 → 1) + Sigmoid → 是否结合
- 回归任务：MLP(256 → 128 → 1) → pKd / pKi 值

两个任务可以做在同一篇论文里，展示模型的通用性。

---

## 三、可解释性分析——论文最核心的章节

这一部分不是附录，是正文的重头戏。我们要让审稿人看到："这个模型不是黑箱，它确实理解了分子识别的机制。"

### 3.1 注意力热图 vs 已知结合位点

**方法：**
1. 选取 PDB 中有共晶结构的药物-靶点对（如 Imatinib—ABL1, Erlotinib—EGFR）
2. 从共晶结构中提取配体 4Å 以内的接触残基作为 ground truth
3. 从模型交互矩阵 M 中提取注意力权重最高的 top-K 残基
4. 计算 Precision@K, Recall@K, F1 — 衡量模型定位结合口袋的准确率

**可视化：**
- 把注意力权重映射到蛋白质 3D 结构上（PyMOL 热力图着色）
- 左边放共晶结构的真实结合口袋，右边放模型预测的注意力热图
- 这张图会是论文里最有冲击力的 Figure

### 3.2 药物亚结构重要性分析

**方法：**
1. 对 GNN 使用 Grad-CAM（梯度加权类激活映射），计算每个原子对预测结果的贡献
2. 将原子级重要性聚合到功能基团级别（如苯环、氨基、羟基等）
3. 与已知的药效团特征对比

**案例分析：**
- 选一个 kinase inhibitor，看模型是否把注意力集中在铰链区（hinge region）结合的杂环上
- 选一个 GPCR 配体，看模型是否识别出氢键锚定基团

### 3.3 蛋白家族层面的交互模式

**方法：**
1. 对同一蛋白家族（如 Kinase）的所有靶点，聚合注意力矩阵
2. 做 clustering / PCA，看是否能自动发现 "DFG-in vs DFG-out" 等已知的构象分类
3. 跨家族比较：Kinase vs GPCR vs Protease，交互模式有什么系统性差异？

**生物学价值：** 这不是在做 ML 消融实验，而是在用计算方法做蛋白质化学生物学分析。

### 3.4 分子对接交叉验证

- 对模型预测为强结合的 drug-target pair，用 DiffDock 做分子对接
- 比较对接 pose 中的接触残基与模型注意力高亮残基的一致性
- 报告 RMSD 和结合自由能（ΔG）

---

## 四、数据集选择——不只是 benchmark

### 4.1 主实验数据集

| 数据集 | 任务类型 | 样本量 | 选择理由 |
|-------|---------|--------|---------|
| **BindingDB** | 二分类 + 回归 | ~2M interactions | 最大规模，涵盖多种靶点家族 |
| **Davis** | 回归 (Kd) | 442 drugs × 379 targets | Kinase 专注，数据质量高 |
| **KIBA** | 回归 (KIBA score) | 2,116 drugs × 229 targets | 综合评分，广泛使用 |

### 4.2 划分策略——体现生物学思维

不做随机划分，做**冷启动划分（Cold-start Split）**：

- **Cold-drug**：测试集中的药物在训练集中从未出现 → 模拟新药筛选
- **Cold-target**：测试集中的靶点在训练集中从未出现 → 模拟新靶点发现
- **Cold-both**：药物和靶点都是全新的 → 最严苛的设定

**为什么这很重要？** 随机划分在现实中没有意义——你永远不会需要预测一个已知 pair 的结合强度。科学家真正需要的是：面对一个全新的蛋白（比如新冠变异株的某个突变体），预测哪些现有药物可能有效。

### 4.3 案例研究数据

从 PDB 中手工挑选 20-30 个有高质量共晶结构的 drug-target complex：
- 确保覆盖主要靶点家族：Kinase, GPCR, Protease, Nuclear Receptor
- 确保有详细的 binding site 注释
- 这些用于可解释性验证，不参与训练

---

## 五、对比方法——不是"比谁更高"，而是"说明什么问题"

### 5.1 Baseline 选择逻辑

每个 baseline 的选择都有明确的目的，用来回答一个特定的科学问题：

| Baseline | 回答的问题 |
|----------|----------|
| **DeepDTA** (CNN-based) | 序列级编码 vs 残基级编码，粒度差异有多大？ |
| **GraphDTA** (GCN/GAT) | 同样用 GNN，加入交互注意力能带来多少提升？ |
| **MolTrans** (Transformer) | Transformer 全局注意力 vs 我们的交叉注意力设计？ |
| **DrugBAN** (Bilinear Attention) | 双线性注意力 vs 多头交叉注意力，哪种交互建模更好？ |
| **Ours w/o Cross-Attn** | 消融：去掉交互模块，退化成简单拼接 |
| **Ours w/o Domain Feat** | 消融：去掉功能域注释，纯数据驱动 |
| **Ours w/o ESM-2** | 消融：用简单 CNN 替代 ESM-2，预训练知识值多少？ |

### 5.2 评价指标

**预测性能（必须报但不是重点）：**
- 分类：AUROC, AUPRC, F1, Precision, Recall
- 回归：MSE, CI (Concordance Index), r²_m

**可解释性评价（论文亮点）：**
- Binding Site Recall@K：模型注意力 top-K 残基与真实结合位点的召回率
- Pharmacophore Hit Rate：模型高亮的药物亚结构中，有多少命中已知药效团
- Attention Consistency：对同一药物与同家族不同靶点的注意力模式一致性

---

## 六、技术实现细节

### 6.1 项目结构

```
BioInteract/
├── configs/                  # 超参数配置
│   └── default.yaml
├── data/
│   ├── raw/                  # 原始数据集
│   ├── processed/            # 处理后的分子图和序列
│   └── esm2_embeddings/      # 预提取的 ESM-2 特征
├── src/
│   ├── data/
│   │   ├── dataset.py        # DTI 数据集类
│   │   ├── mol_graph.py      # SMILES → 分子图（含化学先验特征）
│   │   ├── protein_feat.py   # 蛋白质特征处理 + 功能域注释
│   │   └── split.py          # Cold-start 数据划分
│   ├── models/
│   │   ├── drug_encoder.py   # GINE 药物编码器
│   │   ├── target_encoder.py # ESM-2 投影 + 域特征融合
│   │   ├── interaction.py    # Cross-Attention 交互模块
│   │   ├── biointeract.py    # 完整模型
│   │   └── layers.py         # 通用层（门控池化等）
│   ├── interpret/
│   │   ├── attention_analysis.py   # 注意力热图提取与分析
│   │   ├── gradcam.py              # GNN Grad-CAM
│   │   ├── binding_site_eval.py    # 结合位点预测评估
│   │   └── visualize.py            # PyMOL/matplotlib 可视化
│   └── utils/
│       ├── metrics.py        # 评价指标
│       ├── chemistry.py      # 化学工具函数
│       └── logger.py         # 日志
├── scripts/
│   ├── extract_esm2.py       # ESM-2 特征提取脚本
│   ├── prepare_data.py       # 数据预处理
│   └── dock_validate.py      # DiffDock 对接验证
├── train.py                  # 训练入口
├── evaluate.py               # 评估入口
├── interpret.py              # 可解释性分析入口
├── requirements.txt
└── README.md
```

### 6.2 关键超参数

```yaml
# configs/default.yaml
model:
  drug_encoder:
    gnn_type: GINE
    num_layers: 3
    hidden_dim: 256
    dropout: 0.2
    edge_dim: 16
  
  target_encoder:
    esm2_dim: 1280          # ESM-2 输出维度
    projection_dim: 256     # 投影到的目标维度
    domain_embed_dim: 32    # 功能域 embedding 维度
    num_domain_types: 50    # 域类型数量
  
  interaction:
    num_heads: 8
    attn_dim: 256
    dropout: 0.1
  
  predictor:
    hidden_dims: [256, 128]
    dropout: 0.3

training:
  batch_size: 64
  lr: 1e-4
  weight_decay: 1e-5
  epochs: 100
  patience: 15             # early stopping
  gradient_accumulation: 1 # 如OOM则调大
  amp: true                # 混合精度
  
data:
  dataset: davis           # davis / kiba / bindingdb
  split: cold_drug         # random / cold_drug / cold_target / cold_both
  seed: 42
```

### 6.3 显存预估

| 组件 | 显存占用 | 备注 |
|------|---------|------|
| GINE (3层, 256d) | ~200 MB | 分子图一般 < 100 个原子 |
| 投影层 + 域特征 | ~50 MB | 线性层 |
| Cross-Attention (8 head) | ~500 MB | 取决于序列长度 |
| 预测头 | ~20 MB | MLP |
| 优化器状态 | ~800 MB | Adam |
| 数据 batch | ~400 MB | batch_size=64 |
| **总计** | **~2 GB** | **远低于 16GB 上限** |

结论：显存非常充裕，可以把 batch_size 开到 128 甚至 256，或者增大模型容量。

---

## 七、预期论文结构

```
Title: BioInteract: Interpretable Drug-Target Interaction Prediction 
       via Residue-Level Cross-Attention with Biological Prior Knowledge

Abstract
1. Introduction
   - DTI 预测的生物学意义
   - 现有方法缺乏可解释性
   - 我们的贡献
2. Related Work
3. Methods
   3.1 Problem Formulation
   3.2 Drug Encoder with Chemical Priors
   3.3 Target Encoder with Domain Annotations
   3.4 Residue-Level Cross-Attention
   3.5 Training Objective
4. Experiments
   4.1 Datasets and Cold-Start Splits
   4.2 Prediction Performance (表格)
   4.3 Cold-Start Generalization (关键！)
5. Interpretability Analysis (论文重头戏)
   5.1 Binding Site Localization Accuracy
   5.2 Pharmacophore Recognition
   5.3 Cross-Family Interaction Patterns
   5.4 Case Studies (Imatinib-ABL1, Erlotinib-EGFR, ...)
   5.5 Molecular Docking Cross-Validation
6. Discussion
7. Conclusion
```

---

## 八、执行计划

| 阶段 | 周次 | 交付物 |
|------|------|--------|
| Phase 1: 数据与特征 | W1-W2 | 数据集下载、分子图构建、ESM-2 特征提取 |
| Phase 2: 核心模型 | W3-W4 | BioInteract 完整模型、训练流程通跑 |
| Phase 3: 实验 | W5-W7 | Baseline 对比、消融实验、Cold-start 实验 |
| Phase 4: 可解释性 | W8-W9 | 注意力分析、Grad-CAM、结合位点评估、Case Study |
| Phase 5: 验证 | W10 | DiffDock 对接验证、PyMOL 3D 可视化 |
| Phase 6: 写作 | W11-W12 | 论文初稿 |
