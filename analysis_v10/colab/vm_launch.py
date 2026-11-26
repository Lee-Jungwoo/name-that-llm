# colab exec로 보내는 스크립트: VM 안에서 실행 사본을 papermill로(nohup) 시작합니다. 환경 변수 RUN_NB(확장자 제외), RUN_LOG.
import os, subprocess
nb = os.environ.get("RUN_NB", "v10_smoke")
log = os.environ.get("RUN_LOG", "v10.log")


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr)[-1500:]


# 앞 실행이 남아 있으면 papermill과 그 자식(노트북 커널, GPU를 잡고 있음)을 함께 끝냅니다. 세션 자체의 커널은 건드리지 않습니다.
print(sh("for p in $(pgrep -f '[p]apermill'); do pkill -9 -P $p; kill -9 $p; done; sleep 3; "
         "ps -eo pid,ppid,cmd | grep -E '[p]apermill|[k]ernel_launcher' | cut -c1-110; nvidia-smi --query-gpu=memory.used --format=csv,noheader"))
print(sh("python -c 'import papermill' 2>&1 || pip install -q papermill 2>&1 | tail -2; python -c 'import papermill; print(papermill.__version__)'"))
os.makedirs("/content/out", exist_ok=True)
if os.path.exists(f"/content/out/{log}"):  # 이어 하기: 앞 실행의 로그를 남겨 둠
    os.replace(f"/content/out/{log}", f"/content/out/{log}.{int(os.path.getmtime(f'/content/out/{log}'))}")
# papermill: 셀 출력을 로그로 흘려 보내고(--log-output), 셀이 끝날 때마다 실행된 노트북을 저장합니다.
cmd = (f"cd /content/run && PYTHONUNBUFFERED=1 nohup papermill {nb}.ipynb /content/out/{nb}_executed.ipynb "
       f"--log-output --no-progress-bar --request-save-on-cell-execute > /content/out/{log} 2>&1 &")
subprocess.Popen(cmd, shell=True, start_new_session=True)
print("started", nb)
