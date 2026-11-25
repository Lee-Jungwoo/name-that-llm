#!/bin/bash
# 새 VM에 입력(과 이어 할 파일)을 올리고 LoRA 전용 실행을 시작합니다.
# 사용: REPO=<저장소> S=<작업 폴더> bootstrap.sh <실행 사본 이름(S/run/<이름>.ipynb)> [nolaunch]
# out/v9에 받아 둔 Stage 1 체크포인트·Stage 1 가중치·fold·full 파일이 있으면 /content/out으로 되돌려 놓습니다(RESTORE=0이면 안 함).
T=$(cd $(dirname $0) && pwd); R=$REPO; C=~/.local/bin/colab; SES=${SES:-ntl-v9}; NB=$1; D=$R/out/v9
cd /tmp
q() { grep -v "version\|update\|silence\|^$"; }
up() { for t in 1 2 3; do $C upload -s $SES "$1" "$2" 2>&1 | grep -q Uploaded && { echo -n "1 "; return 0; }; sleep 2; done; echo -n "FAIL($1) "; return 1; }
up_big() { # <local file> <remote path>: 40MB 조각으로 올리고 VM에서 이어 붙인 뒤 md5 비교
  rm -rf $S/parts; mkdir -p $S/parts; (cd $S/parts && split -b 40M -d "$1" up.part.)
  for f in $S/parts/up.part.*; do up $f /content/up/$(basename $f) >/dev/null; done
  vm=$(echo "import subprocess; print('MD5', subprocess.run('cat /content/up/up.part.* > $2 && rm /content/up/up.part.* && md5sum $2', shell=True, capture_output=True, text=True).stdout)" | timeout 150 $C exec -s $SES --timeout 140 2>&1 | grep MD5 | awk '{print $2}')
  loc=$(md5sum "$1" | cut -c1-32); rm -rf $S/parts
  if [ "$vm" = "$loc" ]; then echo "ok $(basename $1) $loc"; else echo "MD5 MISMATCH $(basename $1): vm=$vm local=$loc"; return 1; fi
}
timeout 100 $C exec -s $SES -f $T/vm_setup.py --timeout 90 2>&1 | q | head -8
for f in train.csv test.csv sample_submission.parquet; do up $R/data/name-that-llm/$f /content/data/name-that-llm/$f; done
up $R/analysis_v4/oof_v4.npz /content/run/oof_v4.npz; up $S/run/$NB.ipynb /content/run/$NB.ipynb; up $T/vm_daemon.py /content/run/vm_daemon.py; echo
if [ "${POOL:-upload}" = upload ]; then echo -n "pool: "; up_big $R/data/open-pool/ntl_open_pool.parquet /content/data/open-pool/ntl_open_pool.parquet || exit 1; fi
if [ "${RESTORE:-1}" = 1 ]; then
  if [ -s $D/ntl_lora_v9_stage1.pt ]; then echo -n "resume stage1: "; up_big $D/ntl_lora_v9_stage1.pt /content/out/ntl_lora_v9_stage1.pt || exit 1
  elif [ -s $D/ntl_lora_v9_stage1_ckpt.pt ]; then echo -n "resume stage1 ckpt: "; up_big $D/ntl_lora_v9_stage1_ckpt.pt /content/out/ntl_lora_v9_stage1_ckpt.pt || exit 1; fi
  for k in fold0 fold1 fold2 fold3 fold4 full; do [ -s $D/ntl_lora_v9_$k.pt ] && { echo -n "resume $k: "; up_big $D/ntl_lora_v9_$k.pt /content/out/ntl_lora_v9_$k.pt || exit 1; }; done
fi
echo "import subprocess; subprocess.Popen('cd /content/run && nohup python vm_daemon.py > /content/out/daemon.log 2>&1 &', shell=True, start_new_session=True); print('daemon started')" | timeout 60 $C exec -s $SES --timeout 40 2>&1 | q | tail -1
[ "$2" = nolaunch ] || timeout 200 $C exec -s $SES -f $T/vm_launch.py --env RUN_NB=$NB --env RUN_LOG=v9.log --timeout 180 2>&1 | q | tail -2
date
