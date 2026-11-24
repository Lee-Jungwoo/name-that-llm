#!/bin/bash
# 세션 커널에 1분마다 짧은 exec를 보내 유휴 회수를 막습니다. 세션이 없어지면 끝납니다.
S=$(cd $(dirname $0)/.. && pwd); C=~/.local/bin/colab; cd /tmp; fail=0
while :; do
  out=$(echo 'import subprocess; print("PING", subprocess.run("date +%H:%M:%S; tail -1 /content/out/gpu.log", shell=True, capture_output=True, text=True).stdout.replace("\n", " "))' | timeout 70 $C exec -s ntl-v7 --timeout 45 2>&1 | grep "PING\|not found\|No active")
  echo "[$(date +%H:%M:%S)] ${out:-NO REPLY}" >> $S/logs/pings.log
  if echo "$out" | grep -q PING; then fail=0; else fail=$((fail+1)); fi
  if [ $fail -ge 4 ]; then $C sessions 2>&1 | grep -q "ntl-v7" || { echo "[$(date +%H:%M:%S)] SESSION GONE" | tee -a $S/logs/pings.log; exit 1; }; fi
  [ -f $S/run/STOP_PING ] && exit 0
  sleep 45
done
