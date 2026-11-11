# Colab CLI 세션에서 노트북 실행 전에 한 번 돌리는 준비 스크립트
# 사용법 (Mac 터미널, 프로젝트 폴더에서): colab exec -s ntl -f colab_setup.py --timeout 0
import os

from huggingface_hub import snapshot_download

# 1) 대회 데이터와 결과를 둘 폴더
for d in ["/content/input/name-that-llm", "/content/out"]:
    os.makedirs(d, exist_ok=True)

# 2) Qwen3-4B-Base 내려받기 (약 8GB, 공개 모델이라 토큰 불필요)
snapshot_download("Qwen/Qwen3-4B-Base", local_dir="/content/models/Qwen3-4B-Base")

# 3) 노트북이 읽는 환경 변수 (같은 커널에서 이어지는 exec에 유지됨)
os.environ.update(
    NTL_INPUT="/content/input",
    NTL_BASE_MODEL="/content/models/Qwen3-4B-Base",
    NTL_WORK="/content/out",
    # Open Pool 없이 돌리므로 LoRA 2단계를 처음부터 학습하는 설정으로 바꿈
    NTL_CFG='{"stage2_lr": 2e-4, "stage2_epochs": 4, "stage2_budget_h": 1.0}',
)
print("준비 완료:", {k: v for k, v in os.environ.items() if k.startswith("NTL_")})
