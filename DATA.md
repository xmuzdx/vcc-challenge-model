# 使用的数据

当前 `train.py`、`eval.py`、`submit.py` 只读下面这些已经整理好的文件。`data/public` 里的原始 h5ad 不直接进入训练。

## 训练

源细胞系是 Replogle K562 全基因组筛选 `data/prep/train/gwps.npz`。目标细胞系是：

- `rpe1.npz`
- `hepg2.npz`
- `jurkat.npz`
- `k562ess.npz`（与源细胞系同一 K562 上的另一套筛选）

模型学的是：给定源细胞系上的敲低响应，预测目标细胞系上的 log fold change。

## 本地评估

H1 留出，不参与训练。

- `data/prep/val/h1.npz`：扰动表
- `data/prep/val/h1_eval.h5`：用来重算官方六项（PDS、FID、JAC、NMAE、MSE 和 `score_avg`）

## VCC 2026 提交

- `data/challenge_2026/gene_names.csv`：18533 个基因
- `data/challenge_2026/pert_counts.csv`：300 个靶基因
- `data/challenge_2026/context_A.h5ad`、`context_B.h5ad`、`context_C.h5ad`：三个未见上下文的对照
- `data/prep/test/ctx_A.npz`、`ctx_B.npz`、`ctx_C.npz`：上面三个对照的预处理结果

## 先验

训练时只用三份：

- `prior/gene_priors/full_esm_embeddings.npz`：基因 ESM 嵌入
- `prior/gene_features/go_onto.npz`：GO 的 OPA2Vec
- `prior/pert_classification.tsv`：扰动基因的强度和通路注释

`prior/esm2`、`prior/ppi`、`prior/ppi_graphs`、`prior/gene_sets` 还在磁盘上，当前训练循环没有读它们。
