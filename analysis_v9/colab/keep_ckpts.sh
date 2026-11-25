#!/bin/bash
# fetcher.sh가 최근 두 개만 남기고 지우는 Stage 1 체크포인트를 전부 보관합니다 (fetcher.sh는 건드리지 않음).
# 60초마다 out/v9/ntl_lora_v9_stage1_ckpt.pt* 를 보고, 아직 없는 것을 out/v9/ckpt_history/에 하드링크합니다(안 되면 cp).
# 하드링크는 fetcher가 원본 이름을 지워도 남습니다. fetcher는 .tmp에서 조립해 mv(rename)로 최종 이름을 만들고,
# 이전 ckpt.pt는 ckpt.prev.pt로 rename하므로 같은 파일이 이름만 바꿔 두 번 보일 수 있습니다. 그래서 이름이 아니라 수정 시각(mtime)을 키로 씁니다.
# 받는 중인 파일을 잡지 않도록 크기가 연속 두 번 같을 때만 링크합니다.
# 끝나는 때: v9.log에 "[stage1] 완료"가 있고 새 체크포인트가 10분 동안 없을 때, 또는 시작 12시간 뒤.
# 사용: REPO=<저장소> keep_ckpts.sh   (REPO 기본값: 이 스크립트 기준 저장소 루트)
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
D=$REPO/out/v9; H=$D/ckpt_history; mkdir -p "$H"
declare -A seen   # 파일 경로 -> 직전 확인 때 "크기:mtime"
start=$(date +%s); last_new=$start
echo "[$(date +%H:%M:%S)] 시작 pid $$"
while :; do
  now=$(date +%s)
  for f in "$D"/ntl_lora_v9_stage1_ckpt.pt*; do
    [ -f "$f" ] || continue
    case $f in *.tmp|*.dl|*.assembled) continue;; esac
    sig=$(stat -c '%s:%Y' "$f") || continue
    mt=${sig#*:}; dst=$H/stage1_ckpt_$mt.pt
    if [ "${seen[$f]}" = "$sig" ] && [ ! -e "$dst" ]; then   # 두 번 연속 같음 = 다 받은 파일
      if ln "$f" "$dst" 2>/dev/null || { cp -p "$f" "$dst.part" && mv "$dst.part" "$dst"; }; then
        last_new=$now; echo "[$(date +%H:%M:%S)] 보관: $(basename "$f") -> $(basename "$dst") (${sig%:*} bytes)"
      fi
    fi
    seen[$f]=$sig
  done
  if grep -aq '\[stage1\] 완료' "$D/v9.log" 2>/dev/null && [ $((now - last_new)) -ge 600 ]; then echo "[$(date +%H:%M:%S)] 학습 완료, 새 체크포인트 없음: 종료"; exit 0; fi
  [ $((now - start)) -ge 43200 ] && { echo "[$(date +%H:%M:%S)] 12시간 경과: 종료"; exit 0; }
  sleep 60
done
