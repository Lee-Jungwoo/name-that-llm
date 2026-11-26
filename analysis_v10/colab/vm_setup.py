# colab exec로 보내는 스크립트: VM 사양과 패키지 버전을 찍고 폴더를 만듭니다.
import subprocess


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr)[-3000:]


print(sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | sed -n 2p; df -h /content | tail -1; "
         "python -c 'import torch, transformers, sklearn, numpy; print(\"torch\", torch.__version__, \"transformers\", transformers.__version__, "
         "\"sklearn\", sklearn.__version__, \"numpy\", numpy.__version__)'; "
         "mkdir -p /content/data/name-that-llm /content/data/open-pool /content/out /content/run /content/up /content/dl; ls /content"))
