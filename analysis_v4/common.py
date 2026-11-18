"""v1~v4 예측을 fold 0(지문 540개, 모델당 30개) 기준으로 읽는다.
fold 0은 모든 버전에서 학습에 쓰지 않은 같은 지문 540개다. v1~v3는 fold 0을 뺀 LoRA 하나만 있고,
v4는 5-fold 중 fold 0 모델의 예측을 쓴다. n-gram은 모든 버전이 5-fold라 fold 0 부분만 잘라 쓴다."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VERSIONS = (1, 2, 3, 4)
ORGS = [("GPT", "gpt"), ("Gemini", "gemini"), ("Claude", "claude"), ("Qwen", "qwen"), ("GLM", "glm"), ("Kimi", "kimi")]


def load(v, root=ROOT):
    """(y, 블렌드 예측, n-gram 예측, LoRA 예측, labels). 모두 fold 0 지문 540개."""
    d = np.load(Path(root) / f"oof_v{v}.npz", allow_pickle=True)
    m = d["fold"] == 0
    y = d["y"][m]
    ng = d["oof_ng"][m].argmax(1)
    if v == 4:
        lora = d["lora_oof"][m].argmax(1)
        blend = d["eval_lp"][m].argmax(1)
    else:
        idx = d["eval_idx"]
        assert (np.sort(idx) == np.where(m)[0]).all(), "eval_idx가 fold 0과 다릅니다."
        order = np.argsort(idx)  # fold 0 행 순서에 맞춤
        lora = d["holdout_logits"][order].argmax(1)
        blend = d["eval_lp"][order].argmax(1)
    return y, blend, ng, lora, list(d["labels"])


def recall(y, p, K=18):
    return np.array([(p[y == k] == k).mean() for k in range(K)])


def org_of(labels):
    return [next(n for n, pre in ORGS if l.startswith(pre)) for l in labels]
