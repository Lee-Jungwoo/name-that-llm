"""v7과 v8 제출 커널이 저장한 OOF 로짓으로 라벨별 recall과 fold별 BA를 비교합니다 (CPU, 재학습 없음).

  - v7: analysis_v7/oof_v7.npz, v8: analysis_v8/oof_v8.npz (키 ng_logits, lora_logits, deb_logits, y, ids)
  - 블렌드는 노트북 5장과 같습니다(분기마다 NLL이 가장 작은 온도, 고정 가중치 lora 0.6 / deb 0.2 / ng 0.2).
  - 게이트는 노트북 5-1장과 같은 식을 이 서버의 scikit-learn으로 다시 학습한 값이라 커널 로그와 지문 1개쯤 다를 수 있습니다.
  - fold는 analysis_v4/oof_v4.npz의 fold(= Kaggle 커널의 분할), 차이의 구간은 프롬프트 150개 단위 paired bootstrap 1,000회입니다.
nested 가중치와 n-gram 가중치별 표는 analysis_v7/whatif_ngram.py에 같은 npz를 넣어 계산합니다.

사용: .venv/bin/python analysis_v8/compare_v7.py > analysis_v8/logs/compare_v7.log
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bas

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
          "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
          "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
          "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]
FAMILIES = {p: [i for i, l in enumerate(LABELS) if l.startswith(p)] for p in ("gpt", "gemini", "claude", "glm")}
GATE_ALPHA = dict(gpt=0.75, gemini=0.25, claude=0.5, glm=1.0)
FIXED = dict(lora=0.6, deb=0.2, ng=0.2)
GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3)

v4 = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
y, fold, pid = v4["y"], v4["fold"], v4["prompt_id"]
folds = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(5)]


def lsm(x, t):
    x = x / t
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def best_t(s):
    return GRID[int(np.argmin([-lsm(s, t)[np.arange(len(y)), y].mean() for t in GRID]))]


def gate(lps, blend_lp):
    out = blend_lp.copy()
    for name, cols in FAMILIES.items():
        feat = np.hstack([lp[:, cols] for lp in lps])
        spec = np.zeros((len(y), len(cols)))
        for a, b in folds:
            rows = a[np.isin(y[a], cols)]
            spec[b] = LogisticRegression(C=1.0, max_iter=3000).fit(feat[rows], np.searchsorted(cols, y[rows])).predict_log_proba(feat[b])
        sub = out[:, cols]
        mass = logsumexp(sub, axis=1, keepdims=True)
        new = log_softmax((1 - GATE_ALPHA[name]) * (sub - mass) + GATE_ALPHA[name] * spec, axis=1) + mass
        out[:, cols] = np.where(np.isin(out.argmax(1), cols)[:, None], new, sub)
    return out


def load(path):
    z = np.load(ROOT / path)
    assert (z["y"] == y).all()
    L = {k: lsm(z[f"{k}_logits"], best_t(z[f"{k}_logits"])) for k in ("ng", "lora", "deb")}
    blend = sum(FIXED[k] * L[k] for k in FIXED)
    return dict(ng=z["ng_logits"].argmax(1), lora=z["lora_logits"].argmax(1), deb=z["deb_logits"].argmax(1),
                blend=blend.argmax(1), gate=gate([L["ng"], L["lora"], L["deb"]], blend).argmax(1),
                T={k: best_t(z[f"{k}_logits"]) for k in ("ng", "lora", "deb")})


prompts = np.unique(pid)
rows_of = {p: np.where(pid == p)[0] for p in prompts}
rng = np.random.default_rng(0)
BOOT = [np.concatenate([rows_of[p] for p in rng.choice(prompts, len(prompts))]) for _ in range(1000)]


def paired(b, a):
    d = np.array([bas(y[i], b[i]) - bas(y[i], a[i]) for i in BOOT])
    return f"{bas(y, b) - bas(y, a):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] P>0 {np.mean(d > 0):.2f}"


def recall(pred):
    return np.array([(pred[y == k] == k).mean() for k in range(len(LABELS))])


v7, v8 = load("analysis_v7/oof_v7.npz"), load("analysis_v8/oof_v8.npz")
print("온도:", "v7", v7["T"], "| v8", v8["T"])
assert (v7["lora"] == v8["lora"]).all() and (v7["deb"] == v8["deb"]).all(), "LoRA·DeBERTa OOF가 두 커널에서 다릅니다."
print("LoRA와 DeBERTa의 OOF 예측은 두 커널에서 2,699개 모두 같음")
print("\n== OOF BA (v7 -> v8), fold별, paired bootstrap ==")
for key, name in (("ng", "n-gram 단독"), ("blend", "3분기 블렌드 (고정 가중치)"), ("gate", "게이트 뒤 (로컬 LR)")):
    f7 = [round(float(bas(y[b], v7[key][b])), 4) for _, b in folds]
    f8 = [round(float(bas(y[b], v8[key][b])), 4) for _, b in folds]
    print(f"{name}: {bas(y, v7[key]):.4f} -> {bas(y, v8[key]):.4f} | {paired(v8[key], v7[key])} | 바뀐 예측 {int((v7[key] != v8[key]).sum())}개")
    print(f"  fold별 v7 {f7}\n  fold별 v8 {f8}")
    fixed, broke = int(((v8[key] == y) & (v7[key] != y)).sum()), int(((v8[key] != y) & (v7[key] == y)).sum())
    print(f"  고친 지문 {fixed}개 / 망친 지문 {broke}개")

print("\n== 라벨별 recall (v7 -> v8) ==")
tab = pd.DataFrame(index=LABELS)
for key, name in (("ng", "n-gram"), ("blend", "블렌드"), ("gate", "게이트 뒤")):
    tab[f"{name} v7"], tab[f"{name} v8"] = recall(v7[key]), recall(v8[key])
    tab[f"{name} Δ"] = tab[f"{name} v8"] - tab[f"{name} v7"]
pd.set_option("display.width", 250)
print(tab.round(3).to_string())
print("\n예측 수 (게이트 뒤, 라벨당 정답 150개):")
print(pd.DataFrame({"v7": np.bincount(v7["gate"], minlength=18), "v8": np.bincount(v8["gate"], minlength=18)}, index=LABELS).T.to_string())
