#!/bin/bash
# 새 VM에 입력(과 이어 할 파일)을 올리고 DeBERTa 전용 실행을 시작합니다.
S=$(cd $(dirname $0)/.. && pwd); R=$REPO; C=~/.local/bin/colab; NB=v7_deb; D=$R/out/v7/colab
cd /tmp
q() { grep -v "version\|update\|silence\|^$"; }
up_big() { # <local file> <remote path>: 40MB 조각으로 올리고 VM에서 이어 붙임
  rm -rf $S/parts; mkdir -p $S/parts; (cd $S/parts && split -b 40M -d "$1" up.part.)
  for f in $S/parts/up.part.*; do for t in 1 2 3; do $C upload -s ntl-v7 $f /content/up/$(basename $f) 2>&1 | grep -q Uploaded && break; sleep 2; done; done
  echo "import subprocess; print(subprocess.run('cat /content/up/up.part.* > $2 && rm /content/up/up.part.* && md5sum $2', shell=True, capture_output=True, text=True).stdout)" | timeout 150 $C exec -s ntl-v7 --timeout 140 2>&1 | q | tail -1
  md5sum "$1" | cut -c1-32; rm -rf $S/parts
}
timeout 100 $C exec -s ntl-v7 -f $S/tools/vm_setup.py --timeout 90 2>&1 | q | head -3
echo 'import os; os.makedirs("/content/up", exist_ok=True); os.makedirs("/content/dl", exist_ok=True); print("dirs ok")' | timeout 60 $C exec -s ntl-v7 --timeout 40 2>&1 | q | tail -1
for f in train.csv test.csv sample_submission.parquet; do $C upload -s ntl-v7 $R/data/name-that-llm/$f /content/data/name-that-llm/$f 2>&1 | grep -c Uploaded; done | tr '\n' ' '
$C upload -s ntl-v7 $R/analysis_v4/oof_v4.npz /content/run/oof_v4.npz 2>&1 | grep -c Uploaded
$C upload -s ntl-v7 $S/run/$NB.ipynb /content/run/$NB.ipynb 2>&1 | grep -c Uploaded
$C upload -s ntl-v7 $S/tools/keepalive.py /content/run/keepalive.py 2>&1 | grep -c Uploaded
echo "pool:"; up_big $R/data/open-pool/ntl_open_pool.parquet /content/data/open-pool/ntl_open_pool.parquet
if [ -s $D/ntl_deb_stage1.pt ]; then echo "resume: stage1"; echo "import os; os.makedirs(\"/content/data/deb_resume\", exist_ok=True)" | timeout 60 $C exec -s ntl-v7 --timeout 40 >/dev/null 2>&1; up_big $D/ntl_deb_stage1.pt /content/data/deb_resume/ntl_deb_stage1.pt
elif [ -s $D/deb-stage1_ckpt.pt ]; then echo "resume: stage1 ckpt"; up_big $D/deb-stage1_ckpt.pt /content/out/deb-stage1_ckpt.pt; fi
if [ -s $D/ntl_deb_oof_partial.npz ]; then echo "resume: partial oof"; $C upload -s ntl-v7 $D/ntl_deb_oof_partial.npz /content/out/ntl_deb_oof_partial.npz 2>&1 | grep -c Uploaded; fi
if [ -s $D/v7.log ]; then cp $D/v7.log $D/v7.log.$(date +%H%M%S); fi
timeout 60 $C exec -s ntl-v7 -f $S/tools/runbg.py --env "CMD=python keepalive.py" --env LOGNAME2=keepalive.log --timeout 40 2>&1 | q | tail -1
timeout 150 $C exec -s ntl-v7 -f $S/tools/launch.py --env RUN_NB=$NB --env RUN_LOG=v7.log --timeout 130 2>&1 | q | tail -2
date
