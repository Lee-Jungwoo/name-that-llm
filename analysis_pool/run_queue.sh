#!/bin/bash
# 표본 크기별 실행을 하나씩 차례로 돌립니다. 각 단계는 시작 전에 메모리를 확인하고, 모자라면 건너뜁니다.
# 사용법: bash analysis_pool/run_queue.sh "prepare 3500" "fit n3500" ...
cd "$(dirname "$0")/.."
for step in "$@"; do
  .venv/bin/python -I -W ignore analysis_pool/pool_oof.py $step >> analysis_pool/logs/pool_oof.log 2>> analysis_pool/logs/pool_oof.err
done
