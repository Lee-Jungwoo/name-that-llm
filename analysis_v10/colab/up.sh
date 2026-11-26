#!/bin/bash
# 서버 -> VM 업로드 함수 (bootstrap.sh와 수동 작업에서 source). 큰 파일은 40MB 조각 + VM에서 md5 비교.
# 사용: SES=ntl-v10 S=<작업 폴더> source up.sh; up <local> <remote>; up_big <local> <remote>
C=~/.local/bin/colab; SES=${SES:-ntl-v10}
up() { for t in 1 2 3; do (cd /tmp && $C upload -s $SES "$1" "$2" 2>&1) | grep -q Uploaded && { echo -n "1 "; return 0; }; sleep 2; done; echo -n "FAIL($1) "; return 1; }
up_big() {
  rm -rf $S/parts; mkdir -p $S/parts; (cd $S/parts && split -b 40M -d "$1" up.part.)
  for f in $S/parts/up.part.*; do up $f /content/up/$(basename $f) >/dev/null || return 1; done
  vm=$(echo "import subprocess; print('MD5', subprocess.run('mkdir -p \$(dirname $2) && cat /content/up/up.part.* > $2 && rm /content/up/up.part.* && md5sum $2', shell=True, capture_output=True, text=True).stdout)" | (cd /tmp && timeout 200 $C exec -s $SES --timeout 190 2>&1) | grep MD5 | awk '{print $2}')
  loc=$(md5sum "$1" | cut -c1-32); rm -rf $S/parts
  if [ "$vm" = "$loc" ]; then echo "ok $(basename $1) $loc"; else echo "MD5 MISMATCH $(basename $1): vm=$vm local=$loc"; return 1; fi
}
vmexec() { (cd /tmp && timeout ${2:-120} $C exec -s $SES --timeout $(( ${2:-120} - 10 )) 2>&1) <<< "$1" | grep -v "version\|update\|silence\|^$"; }
