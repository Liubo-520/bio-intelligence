# BioInteract: Multi-Split Evaluation Results Summary

## 1. Model Architecture

**BioInteract** — A multimodal deep learning framework for Drug-Target Interaction prediction.

| Component | Architecture | Dimensions |
|-----------|-------------|------------|
| Drug Encoder | GINE GNN (3 layers) | 256d hidden |
| Target Encoder | ESM-2 (150M) + Projection | 640d → 256d |
| Interaction Module | 8-head Cross-Attention + Gated Pooling | 256d |
| Predictor | MLP (256 → 128 → 1) | Dropout 0.3 |

- **Total Parameters**: 2,442,083
- **Drug Features**: 52-dim atom features, 16-dim edge features
- **Target Features**: ESM-2 150M pre-trained embeddings (native 640d)

---

## 2. Dataset & Experimental Setup

| Item | Detail |
|------|--------|
| Dataset | Davis Kinase Binding |
| Interactions | 30,056 (68 drugs × 442 targets) |
| Positive Ratio | ~5% (Kd < 30nM threshold) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | Cosine Annealing (5 warmup epochs) |
| Batch Size | 64 |
| Max Epochs | 100 |
| Early Stopping | Patience=15 (val AUROC) |
| Mixed Precision | AMP (FP16) |
| Hardware | NVIDIA RTX 4080 (16GB VRAM) |
| Seed | 42 |

---

## 3. Multi-Split Test Results

### 3.1 Main Results Table

| Metric | Random | Cold Target | Cold Drug |
|--------|--------|-------------|-----------|
| **AUROC** | **0.9260** | **0.9298** | **0.7251** |
| **AUPRC** | 0.5942 | 0.5155 | 0.1248 |
| **F1** | 0.5782 | 0.5182 | 0.1726 |
| **Precision** | 0.4876 | 0.3910 | 0.2282 |
| **Recall** | 0.7101 | 0.7680 | 0.1388 |
| Test Loss | 1.1310 | 0.7481 | 3.5050 |

### 3.2 Training Summary

| Split | Train / Val / Test | Best Epoch | Best Val AUROC | Early Stop Epoch | Epoch Time |
|-------|-------------------|------------|----------------|------------------|------------|
| Random | 21,040 / 3,005 / 6,011 | 24 | 0.9532 | 39 | ~35s |
| Cold Target | 21,080 / 2,992 / 5,984 | 19 | 0.9065 | 34 | ~32s |
| Cold Drug | 21,658 / 2,652 / 5,746 | 19 | 0.6327 | 34 | ~32s |

---

## 4. Analysis

### 4.1 Random Split
最优表现。训练集与测试集的药物和靶标均有重叠，AUROC达到0.926，验证集AUROC峰值0.9532。模型在第24轮达到最佳，第39轮早停。

**训练曲线关键点**:
- E001: val_auroc=0.8189 → E007: 0.9006 (快速提升)
- E016: 0.9479 → E024: 0.9532 (最佳)
- E025-E039: 波动下降，触发早停

### 4.2 Cold Target Split
对未见过靶标的泛化性能优异。AUROC=0.9298，甚至略高于Random split，表明ESM-2蛋白质编码器为未知靶标提供了优秀的表示能力。Recall最高(0.768)，说明模型对新靶标的正样本检出率强。

**训练曲线关键点**:
- E001: val_auroc=0.8138 → E007: 0.8767 (稳步上升)
- E016: 0.9029 → E019: 0.9065 (最佳)
- E020-E034: 在0.88-0.90之间波动，E034早停

### 4.3 Cold Drug Split
对未见过药物的泛化是最大挑战，AUROC=0.7251。验证集AUROC仅达到0.6327，且训练过程不稳定（val_auroc在0.50-0.63之间大幅波动）。这反映了小分子结构空间的复杂性——仅68种药物训练数据不足以覆盖新药物的化学多样性。

**训练曲线关键点**:
- E001: val_auroc=0.4080 → E010: 0.6072
- E019: 0.6327 (最佳) → 训练过程震荡严重
- 验证集loss持续增大(0.78→2.31)，严重过拟合

---

## 5. Key Findings

1. **ESM-2蛋白质编码器泛化能力突出**：Cold Target AUROC(0.9298)几乎与Random(0.9260)持平，证明ESM-2 150M的蛋白质表示具有强泛化能力，即使面对训练中未见过的靶标蛋白也能产生有效嵌入。

2. **Cold Drug是主要瓶颈**：Cold Drug场景下性能显著下降(AUROC=0.7251)，这与领域共识一致——当药物数据有限(仅68种)时，GNN难以学到足够泛化的分子表示。

3. **模型训练效率高**：所有split在19-24个epoch即达到最佳，利用AMP float16训练，单epoch仅需32-35秒(RTX 4080)。

4. **类别不平衡影响**：~5%正样本率导致Precision偏低（随机split仅0.49），但AUROC/AUPRC更适合评估排序性能。

---

## 6. Interpretability Analysis (Cold Drug Split)

已完成10个案例的可解释性分析，涵盖3个重要激酶靶标：

| Case Study | Drug | Target | Key Findings |
|------------|------|--------|-------------|
| 1-4 | Imatinib, Erlotinib, Sorafenib, Dasatinib | ABL1 | 注意力热图定位到激酶结构域催化位点残基 |
| 5-7 | Gefitinib, Lapatinib, Vandetanib | EGFR | 交叉注意力聚焦于ATP结合口袋关键残基 |
| 8-10 | Vemurafenib, Dabrafenib, Trametinib | BRAF | GradCAM高亮药物分子的药效团子结构 |

- 输出: 30张可视化图(PNG), JSON分析报告
- 路径: `results/interpretability/`

---

## 7. File Inventory

### Checkpoints
| File | Split | Size |
|------|-------|------|
| `checkpoints/best.pt` | Cold Drug | ~29 MB |
| `checkpoints/best_random.pt` | Random | ~9.8 MB |
| `checkpoints/best_cold_target.pt` | Cold Target | ~9.8 MB |

### Results
| File | Description |
|------|-------------|
| `results/test_random.json` | Random split test metrics |
| `results/test_cold_target.json` | Cold target split test metrics |
| `results/experiments.json` | Cold drug experiment record (full config + metrics) |
| `results/interpretability/` | 30 PNG figures + JSON analysis report |

### Training Logs
| File | Description |
|------|-------------|
| `logs/run_random.log` | Random split training log (39 epochs) |
| `logs/run_cold_target.log` | Cold target training log (34 epochs) |
| `logs/train_20260211_170236.log` | Cold drug training log (34 epochs) |

### Source Code
| File | Description |
|------|-------------|
| `train.py` | Original training script |
| `run_split.py` | Multi-split evaluation training script |
| `interpret.py` | Interpretability analysis script |
| `configs/default.yaml` | Hyperparameter configuration |
| `src/` | Model modules (data, model, utils) |

---

## 8. Comparison with Literature

| Method | Davis AUROC (Random) | Reference |
|--------|---------------------|-----------|
| DeepDTA (2018) | 0.878 | Öztürk et al. |
| GraphDTA (2020) | 0.893 | Nguyen et al. |
| MolTrans (2021) | 0.907 | Huang et al. |
| DrugBAN (2022) | 0.915 | Bai et al. |
| **BioInteract (Ours)** | **0.926** | — |

> 注：文献对比仅作参考，不同研究的实验设置(阈值、数据划分、预处理)可能存在差异。

---

*Generated: 2026-02-11 | BioInteract Multi-Split Evaluation*
