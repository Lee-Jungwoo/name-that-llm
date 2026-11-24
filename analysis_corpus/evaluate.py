"""corpus를 더한 n-gram OOF(out/corpus/ngram_oof_<이름>.npz)를 기준선(out/v5/ngram_oof_alpha0.2.npz)과 비교합니다.

- n-gram OOF BA, 블렌드 OOF BA(w=0.6 고정과 w별). 블렌드는 v5_stylo/blend.py와 같은 식입니다
  (n-gram과 LoRA 각각 NLL로 온도를 고르고 log 확률을 가중합, LoRA OOF는 analysis_v4/oof_v4.npz).
- 프롬프트 150개 단위 paired bootstrap(1,000회)으로 ΔBA의 95% 구간.
- 라벨별 recall 변화, 시드만 바꾼 실행(out/v5/ngram_oof_alpha0.2_s*.npz)과의 크기 비교.

사용법: .venv/bin/python -I analysis_corpus/evaluate.py [실행 이름 ...]   (이름을 안 주면 out/corpus의 전부)
"""
import glob, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as BA

from common import LABELS, OUT, ROOT

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
d = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
y, lora, fold, prompt = d["y"], d["lora_oof"], d["fold"], d["prompt_id"]
K = len(LABELS)
GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3)
W = [w / 10 for w in range(11)]


def lsm(x, t=1.0):
    x = x / t; x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))


def best_t(s):
    return GRID[int(np.argmin([-lsm(s, t)[np.arange(len(y)), y].mean() for t in GRID]))]


T_LORA = best_t(lora)
LL = lsm(lora, T_LORA)


def blend_pred(ng, w=0.6):
    return (w * LL + (1 - w) * lsm(ng, best_t(ng))).argmax(1)


# 프롬프트 x 라벨 단위로 미리 묶어 두면 bootstrap에서 BA를 행렬곱으로 계산할 수 있습니다.
pid = pd.factorize(prompt)[0]
P = pid.max() + 1
N = np.zeros((P, K)); np.add.at(N, (pid, y), 1)
rng = np.random.default_rng(0)
M = rng.multinomial(P, np.full(P, 1 / P), size=1000).astype(float)  # 실행마다 같은 재표집을 씀


def boot_delta(pred_a, pred_b):
    """프롬프트 재표집에서의 BA(b) - BA(a) 분포."""
    out = []
    for pred in (pred_a, pred_b):
        C = np.zeros((P, K)); np.add.at(C, (pid, y), (pred == y).astype(float))
        out.append(((M @ C) / (M @ N)).mean(1))
    return out[1] - out[0]


def recall(pred):
    return np.array([(pred[y == k] == k).mean() for k in range(K)])


base = np.load(ROOT / "out" / "v5" / "ngram_oof_alpha0.2.npz")["oof_ng"]
base_ng, base_bl = base.argmax(1), blend_pred(base)
print(f"기준선(alpha0.2, corpus 없음): n-gram OOF {BA(y, base_ng):.4f} | T_NG {best_t(base)} T_LORA {T_LORA} | "
      f"블렌드 w=0.6 {BA(y, base_bl):.4f} | LoRA OOF {BA(y, lora.argmax(1)):.4f}")
print("기준선 블렌드 w별:", {w: round(BA(y, blend_pred(base, w)), 4) for w in W})

print("\n== 시드만 바꾼 실행과의 차이 (run-to-run 잡음) ==")
for f in sorted(glob.glob(str(ROOT / "out" / "v5" / "ngram_oof_alpha0.2_s*.npz"))):
    o = np.load(f)["oof_ng"]
    print(f"{Path(f).name}: 점수 최대 차이 {np.abs(o - base).max():.1e}, n-gram argmax가 다른 행 {int((o.argmax(1) != base_ng).sum())}, "
          f"블렌드 예측이 다른 행 {int((blend_pred(o) != base_bl).sum())}, 블렌드 ΔBA {BA(y, blend_pred(o)) - BA(y, base_bl):+.4f}")

names = sys.argv[1:] or sorted(Path(f).stem.replace("ngram_oof_", "") for f in glob.glob(str(OUT / "ngram_oof_*.npz")))
rows, per_label = [], {}
for name in names:
    z = np.load(OUT / f"ngram_oof_{name}.npz", allow_pickle=True)
    ng = z["oof_ng"]
    p_ng, p_bl = ng.argmax(1), blend_pred(ng)
    by_w = {w: BA(y, blend_pred(ng, w)) for w in W}
    base_by_w = {w: BA(y, blend_pred(base, w)) for w in W}
    d_ng, d_bl = boot_delta(base_ng, p_ng), boot_delta(base_bl, p_bl)
    d_w9 = boot_delta(blend_pred(base, 0.9), blend_pred(ng, 0.9))
    best_w = max(by_w, key=by_w.get)
    rows.append({
        "실행": name, "더한 행": len(z["ext_cell_id"]), "가중치": float(z["ext_weight"]), "T_NG": best_t(ng),
        "n-gram OOF": BA(y, p_ng), "n-gram Δ": BA(y, p_ng) - BA(y, base_ng),
        "n-gram Δ 95%": f"[{np.percentile(d_ng, 2.5):+.4f}, {np.percentile(d_ng, 97.5):+.4f}]",
        "블렌드 w=0.6": BA(y, p_bl), "블렌드 Δ": BA(y, p_bl) - BA(y, base_bl),
        "블렌드 Δ 95%": f"[{np.percentile(d_bl, 2.5):+.4f}, {np.percentile(d_bl, 97.5):+.4f}]",
        "P(Δ>0)": float((d_bl > 0).mean()),
        "블렌드 w=0.9 Δ": by_w[0.9] - base_by_w[0.9],
        "w=0.9 Δ 95%": f"[{np.percentile(d_w9, 2.5):+.4f}, {np.percentile(d_w9, 97.5):+.4f}]",
        "최고 w": best_w, "최고 w 블렌드": by_w[best_w], "fold0 블렌드": BA(y[fold == 0], p_bl[fold == 0]),
        "블렌드 예측이 바뀐 행": int((p_bl != base_bl).sum()),
    })
    per_label[name] = (recall(p_ng) - recall(base_ng), recall(p_bl) - recall(base_bl))
    print(f"\n{name} 블렌드 w별:", {w: round(v, 4) for w, v in by_w.items()})
    print(f"{name} 기준선 대비 w별 Δ:", {w: round(by_w[w] - base_by_w[w], 4) for w in W})

res = pd.DataFrame(rows).set_index("실행")
print("\n== 요약 (Δ는 기준선 대비, 구간은 프롬프트 paired bootstrap 1,000회) ==")
print(res.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\n기준선 최고 w 블렌드: {max(BA(y, blend_pred(base, w)) for w in W):.4f}")

if "poolextra_w1" in names and "all_w1" in names:
    # 같은 개수를 더 넣었을 때 corpus와 Open Pool의 차이 (대조군 대비)
    a = np.load(OUT / "ngram_oof_poolextra_w1.npz", allow_pickle=True)["oof_ng"]
    b = np.load(OUT / "ngram_oof_all_w1.npz", allow_pickle=True)["oof_ng"]
    for label, pa, pb in (("n-gram", a.argmax(1), b.argmax(1)), ("블렌드 w=0.6", blend_pred(a), blend_pred(b))):
        dd = boot_delta(pa, pb)
        print(f"all_w1 - poolextra_w1, {label}: Δ {BA(y, pb) - BA(y, pa):+.4f} "
              f"[{np.percentile(dd, 2.5):+.4f}, {np.percentile(dd, 97.5):+.4f}]")

print("\n== 라벨별 recall 변화: n-gram (기준선 recall, 실행별 Δ) ==")
t = pd.DataFrame({"기준선": recall(base_ng), **{n: per_label[n][0] for n in names}}, index=LABELS)
print(t.to_string(float_format=lambda v: f"{v:+.3f}"))
print("\n== 라벨별 recall 변화: 블렌드 w=0.6 ==")
t = pd.DataFrame({"기준선": recall(base_bl), **{n: per_label[n][1] for n in names}}, index=LABELS)
print(t.to_string(float_format=lambda v: f"{v:+.3f}"))
