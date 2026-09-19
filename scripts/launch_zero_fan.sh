#!/usr/bin/env bash
# 八卡 DDP 合训一个 zero_fan。先 H1 留出，再 --final 并入 H1。
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/data/lyw/ag_final/envs/vcc/bin/python}"
EPOCHS=${EPOCHS:-80}
FINAL_EPOCHS=${FINAL_EPOCHS:-8}

mkdir -p logs runs

echo "[launch] cellpool (skip if present)"
"$PY" data/cellpool.py --lines rpe1 hepg2 jurkat k562ess h1 \
    > logs/cellpool.log 2>&1 || true

echo "[launch] DDP train zero_fan (H1 held out)"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    run.py train --preset zero_fan --use-ddp --gpu 0 \
    --epochs "$EPOCHS" --tag zero_fan \
    > logs/train_zero_fan.log 2>&1

echo "[launch] --final retrain including H1"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    run.py train --preset zero_fan_final --use-ddp --gpu 0 \
    --final --final-epochs "$FINAL_EPOCHS" --tag zero_fan_final \
    > logs/train_zero_fan_final.log 2>&1

echo "[launch] write submission"
"$PY" run.py submit --preset zero_fan_final --write --tta \
    --ckpt zero_fan_final --submit-tag zero_fan_v1 \
    > logs/submit_zero_fan.log 2>&1

echo "done"
