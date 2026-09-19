#!/usr/bin/env bash
# 八卡并行训练：通过 run.py + launch8 预设启动。
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/data/lyw/ag_final/envs/vcc/bin/python}"
BUDGET=${BUDGET:-11400}
EPOCHS=${EPOCHS:-300}

mkdir -p logs runs

for g in 0 1 2 3 4 5 6 7; do
  tag="g${g}"
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 nohup "$PY" run.py train \
      --preset "launch8_g${g}" --gpu 0 \
      --epochs "$EPOCHS" --time-budget "$BUDGET" \
      > "logs/train_${tag}.log" 2>&1 &
  echo "launched $tag on gpu $g (preset=launch8_g${g})"
  sleep 6
done
wait
echo "all runs finished"
