#!/bin/bash
# VM의 결과를 계속 받아 둡니다: 로그와 npz는 그대로, .pt는 /content/dl의 조각 스냅샷을 받아 md5 확인 후 이어 붙입니다.
# Stage 1 체크포인트는 최근 두 개(…_ckpt.pt, …_ckpt.prev.pt)를 남깁니다.
# 사용: REPO=<저장소> S=<작업 폴더> fetcher.sh   (S/run/STOP_FETCH 파일이 생기면 끝남)
C=~/.local/bin/colab; SES=${SES:-ntl-v10}; D=$REPO/out/v10; mkdir -p $D $D/.tmp; cd /tmp
dl() { timeout 150 $C download -s $SES "$1" "$2.dl" >/dev/null 2>&1 && [ -s "$2.dl" ] && mv "$2.dl" "$2"; }
while :; do
  [ -f $S/run/STOP_FETCH ] && exit 0
  for f in v10.log gpu.log deb_curve.csv ntl_deb_v10_oof_partial.npz ntl_deb_v10_oof.npz v10_deb_executed.ipynb daemon.log; do dl /content/out/$f $D/$f; done
  for snap in $(timeout 60 $C ls -s $SES /content/dl 2>/dev/null | grep "\.pt\.[0-9]*/\?$" | tr -d '/'); do
    name=${snap%.*}
    [ -f $D/.tmp/$snap.fetched ] && continue
    timeout 60 $C ls -s $SES /content/dl/$snap 2>/dev/null | grep -q "^DONE" || continue
    T=$D/.tmp/$snap; mkdir -p $T
    dl /content/dl/$snap/nparts $T/nparts && dl /content/dl/$snap/md5 $T/md5 || continue
    ok=1; for i in $(seq 0 $(( $(cat $T/nparts) - 1 ))); do p=$(printf "part.%02d" $i); [ -s $T/$p ] || dl /content/dl/$snap/$p $T/$p || { ok=0; break; }; done
    [ $ok = 1 ] || { rm -rf $T; continue; }
    cat $T/part.* > $D/.tmp/$name.assembled
    if [ "$(md5sum $D/.tmp/$name.assembled | cut -d' ' -f1)" = "$(cat $T/md5)" ]; then
      case $name in *_ckpt.pt) [ -s $D/$name ] && mv $D/$name $D/${name%.pt}.prev.pt;; esac
      mv $D/.tmp/$name.assembled $D/$name; touch $D/.tmp/$snap.fetched
      echo "[$(date +%H:%M:%S)] fetched $snap ($(stat -c %s $D/$name) bytes)" >> $D/fetch.log
    else echo "[$(date +%H:%M:%S)] md5 mismatch $snap" >> $D/fetch.log; fi
    rm -rf $T
  done
  sleep 25
done
