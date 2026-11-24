#!/bin/bash
# VM의 결과를 계속 받아 둡니다: 작은 파일은 그대로, 큰 .pt는 /content/dl의 조각 스냅샷을 받아 md5 확인 후 이어 붙입니다.
S=$(cd $(dirname $0)/.. && pwd); C=~/.local/bin/colab; D=$REPO/out/v7/colab; mkdir -p $D $D/ntl_deb_base $D/.tmp
dl() { timeout 100 $C download -s ntl-v7 "$1" "$2.dl" >/dev/null 2>&1 && [ -s "$2.dl" ] && mv "$2.dl" "$2"; }
while :; do
  [ -f $S/run/STOP_FETCH ] && exit 0
  for f in v7.log gpu.log ntl_deb_oof_partial.npz ntl_deb_oof.npz v7_deb_executed.ipynb; do dl /content/out/$f $D/$f; done
  for f in config.json tokenizer.json tokenizer_config.json spm.model special_tokens_map.json added_tokens.json; do [ -s $D/ntl_deb_base/$f ] || dl /content/out/ntl_deb_base/$f $D/ntl_deb_base/$f; done
  for snap in $(timeout 60 $C ls -s ntl-v7 /content/dl 2>/dev/null | grep "\.pt\.[0-9]*/\?$" | tr -d '/'); do
    name=${snap%.*}
    [ -f $D/.tmp/$snap.fetched ] && continue
    timeout 60 $C ls -s ntl-v7 /content/dl/$snap 2>/dev/null | grep -q "^DONE" || continue
    T=$D/.tmp/$snap; mkdir -p $T
    dl /content/dl/$snap/nparts $T/nparts && dl /content/dl/$snap/md5 $T/md5 || continue
    ok=1; for i in $(seq 0 $(( $(cat $T/nparts) - 1 ))); do p=$(printf "part.%02d" $i); [ -s $T/$p ] || dl /content/dl/$snap/$p $T/$p || { ok=0; break; }; done
    [ $ok = 1 ] || { rm -rf $T; continue; }
    cat $T/part.* > $D/.tmp/$name.assembled
    if [ "$(md5sum $D/.tmp/$name.assembled | cut -d' ' -f1)" = "$(cat $T/md5)" ]; then mv $D/.tmp/$name.assembled $D/$name; touch $D/.tmp/$snap.fetched; echo "[$(date +%H:%M:%S)] fetched $snap ($(stat -c %s $D/$name) bytes)" >> $D/fetch.log; else echo "[$(date +%H:%M:%S)] md5 mismatch $snap" >> $D/fetch.log; fi
    rm -rf $T
  done
  sleep 25
done
