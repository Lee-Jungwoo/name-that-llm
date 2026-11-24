"""두 n-gram OOF(npz)의 차이를 프롬프트 150개 단위 bootstrap으로 봅니다 (n-gram 단독, 블렌드 w = 0.6 / 0.8).

블렌드는 blend.py와 같습니다(v4 LoRA OOF, n-gram 온도 0.2, LoRA 온도 1).
사용법: python v5_stylo/compare_oof.py <기준 npz> <비교할 npz>
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score as BA

root = Path(__file__).resolve().parent.parent
a_name, b_name = sys.argv[1:3]
d = np.load(f"{root}/analysis_v4/oof_v4.npz", allow_pickle=True)
y, lora, pid = d["y"], d["lora_oof"], d["prompt_id"]


def lsm(x, t=1.0):
    x = x / t; x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))


ll = lsm(lora, 1.0)
A = np.load(a_name)["oof_ng"]
B = np.load(b_name)["oof_ng"]
preds = {}
for nm, o in (("a", A), ("b", B)):
    preds[nm, "ng"] = o.argmax(1)
    for w in (0.6, 0.8):
        preds[nm, w] = (w * ll + (1 - w) * lsm(o, 0.2)).argmax(1)
prompts = np.unique(pid)
rows = {p: np.where(pid == p)[0] for p in prompts}
rng = np.random.default_rng(0)
out = {k: [] for k in ("ng", 0.6, 0.8)}
for _ in range(1000):
    idx = np.concatenate([rows[p] for p in rng.choice(prompts, len(prompts))])
    for k in out:
        out[k].append(BA(y[idx], preds["b", k][idx]) - BA(y[idx], preds["a", k][idx]))
print(f"{b_name} - {a_name} (프롬프트 bootstrap 1,000회)")
for k in out:
    v = np.array(out[k]); pt = BA(y, preds["b", k]) - BA(y, preds["a", k])
    n_diff = int((preds["a", k] != preds["b", k]).sum())
    print(f"  {str(k):4s}: 차이 {pt:+.4f}, 95% 구간 [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}], P(차이>0) {np.mean(v > 0):.2f}, 예측이 바뀐 지문 {n_diff}개")
