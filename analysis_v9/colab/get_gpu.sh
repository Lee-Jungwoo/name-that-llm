#!/bin/bash
# GPU 세션을 잡을 때까지 순서대로 요청합니다. 성공하면 종류를 출력하고 끝납니다. 사용: GPUS="L4 A100 L4" get_gpu.sh
C=~/.local/bin/colab
i=0
for gpu in ${GPUS:-L4 A100 L4}; do
  i=$((i+1))
  echo "[$(date +%H:%M:%S)] try $i: $gpu"
  COLAB_CLI_REQUEST_TIMEOUT=330 COLAB_CLI_ASSIGN_RETRIES=0 $C new -s ntl-v9 --gpu $gpu 2>&1 | grep -E "READY|Bad Gateway|refused|Error|error|40[0-9]|50[0-9]" | head -3
  if $C sessions 2>&1 | grep -q "ntl-v9"; then
    echo "[$(date +%H:%M:%S)] OK $gpu"; $C sessions 2>&1 | grep -v "version\|update\|silence"; exit 0
  fi
  $C sessions 2>&1 | grep "\[?\]"   # 고아 세션 확인
  sleep 20
done
echo "[$(date +%H:%M:%S)] FAILED all tries"; exit 1
