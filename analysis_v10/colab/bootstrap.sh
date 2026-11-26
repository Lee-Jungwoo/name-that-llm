#!/bin/bash
# 새 VM에 입력(과 이어 할 파일)을 올리고 DeBERTa 전용 실행을 시작합니다 (v10).
# 사용: REPO=<저장소> S=<작업 폴더> [STOP_AT=<유닉스 시각>] bootstrap.sh <실행 사본 이름(S/run/<이름>.ipynb)> [nolaunch]
# Open Pool은 항상 서버의 고정판(md5 10d1c13b…)을 올립니다(준비 셀이 md5를 확인하고, 원본을 받지 않음).
# out/v10에 받아 둔 Stage 1 가중치(없으면 체크포인트), fold OOF 부분 파일, 전체 학습 모델, deb_curve.csv가 있으면
# /content/out으로 되돌려 놓습니다(RESTORE=0이면 안 함).
T=$(cd $(dirname $0) && pwd); R=$REPO; SES=${SES:-ntl-v10}; NB=$1; D=$R/out/v10
source $T/up.sh
cd /tmp
q() { grep -v "version\|update\|silence\|^$"; }
timeout 100 $C exec -s $SES -f $T/vm_setup.py --timeout 90 2>&1 | q | head -8
for f in train.csv test.csv sample_submission.parquet; do up $R/data/name-that-llm/$f /content/data/name-that-llm/$f; done
up $R/analysis_v4/oof_v4.npz /content/run/oof_v4.npz; up $S/run/$NB.ipynb /content/run/$NB.ipynb; up $T/vm_daemon.py /content/run/vm_daemon.py; echo
echo -n "pool: "; up_big $R/data/open-pool/ntl_open_pool.parquet /content/data/open-pool/ntl_open_pool.parquet || exit 1
if [ "${RESTORE:-1}" = 1 ]; then
  if [ -s $D/ntl_deb_v10_stage1.pt ]; then echo -n "resume stage1: "; up_big $D/ntl_deb_v10_stage1.pt /content/out/ntl_deb_v10_stage1.pt || exit 1
  elif [ -s $D/ntl_deb_v10_stage1_ckpt.pt ]; then echo -n "resume stage1 ckpt: "; up_big $D/ntl_deb_v10_stage1_ckpt.pt /content/out/ntl_deb_v10_stage1_ckpt.pt || exit 1; fi
  [ -s $D/ntl_deb_v10_full.pt ] && { echo -n "resume full: "; up_big $D/ntl_deb_v10_full.pt /content/out/ntl_deb_v10_full.pt || exit 1; }
  for f in ntl_deb_v10_oof_partial.npz deb_curve.csv; do [ -s $D/$f ] && { echo -n "resume $f: "; up $D/$f /content/out/$f; echo; }; done
fi
echo "import subprocess; subprocess.Popen('cd /content/run && nohup python vm_daemon.py > /content/out/daemon.log 2>&1 &', shell=True, start_new_session=True); print('daemon started')" | timeout 60 $C exec -s $SES --timeout 40 2>&1 | q | tail -1
[ "$2" = nolaunch ] || timeout 200 $C exec -s $SES -f $T/vm_launch.py --env RUN_NB=$NB --env RUN_LOG=v10.log ${STOP_AT:+--env NTL_STOP_AT=$STOP_AT} --timeout 180 2>&1 | q | tail -2
date
