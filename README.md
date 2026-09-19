# vcc-challenge-model

VCC 扰动响应预测模型代码包（SMP 风格结构），用于 lab146 / lab130 双环境同步实验进展。

## 目录结构

```
model/
├── config/       # 实验预设、损失权重、checkpoint 注册表
├── data/         # 数据加载与预处理
├── engine/       # 训练引擎、采样、DDP
├── experiment/   # 实验编排
├── inference/    # 推理、提交、TTA
├── losses/       # 损失函数（six_score 等）
├── metrics/      # 官方评测指标
├── models/       # 模型定义（pert_response, zero_fan 等）
├── scripts/      # 启动脚本（launch8.sh 等）
├── tools/        # 诊断与辅助工具
├── run.py        # 统一 CLI 入口
└── paths.py      # 路径配置
```

## 快速开始

```bash
# 查看可用预设
python run.py catalog

# 评估
python run.py eval --preset final

# 单卡训练
python run.py train --preset launch8_g0 --gpu 0

# 八卡并行
bash scripts/launch8.sh
```

## 数据路径

代码默认数据目录为 `model/` 的上一级 `data/`（即 `vcc_challenge/data/`）。
checkpoint 与运行日志写入本地 `runs/`、`logs/`（已在 `.gitignore` 中排除）。

## 双环境同步

```bash
# 拉取最新代码
git pull

# 推送实验代码更新
git add -A && git commit -m "your message" && git push
```
