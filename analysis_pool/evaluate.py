"""Open Pool 표본 크기별 n-gram OOF(out/pool/ngram_oof_<이름>.npz)를 기준선(out/v5/ngram_oof_alpha0.2.npz)과 비교합니다.

- n-gram OOF BA, 블렌드 OOF BA(w=0.6 고정과 w별). 블렌드는 v5_stylo/blend.py와 같은 식입니다
  (n-gram과 LoRA 각각 NLL로 온도를 고르고 log 확률을 가중합, LoRA OOF는 analysis_v4/oof_v4.npz).
- 프롬프트 150개 단위 paired bootstrap(1,000회)으로 ΔBA의 95% 구간. 크기별 Δ의 평균(추세)도 같은 재표집으로 냅니다.
- 표본 크기별 비용(벡터화·fit 시간, 최대 메모리)과 train에서 멀어진 정도(out/pool/cache_n<n>/meta.json).
- 라벨별 recall 변화.

사용법: .venv/bin/python -I analysis_pool/evaluate.py [실행 이름 ...]   (이름을 안 주면 out/pool의 전부)
"""
import glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis_corpus"))
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as BA

from common import LABELS, ROOT

OUT = ROOT / "out" / "pool"
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


pid = pd.factorize(prompt)[0]
P = pid.max() + 1
N = np.zeros((P, K)); np.add.at(N, (pid, y), 1)
M = np.random.default_rng(0).multinomial(P, np.full(P, 1 / P), size=1000).astype(float)  # 실행마다 같은 재표집


def boot_ba(pred):
    C = np.zeros((P, K)); np.add.at(C, (pid, y), (pred == y).astype(float))
    return ((M @ C) / (M @ N)).mean(1)


def ci(x):
    return f"[{np.percentile(x, 2.5):+.4f}, {np.percentile(x, 97.5):+.4f}]"


def recall(pred):
    return np.array([(pred[y == k] == k).mean() for k in range(K)])


base = np.load(ROOT / "out" / "v5" / "ngram_oof_alpha0.2.npz")["oof_ng"]
base_ng, base_bl = base.argmax(1), blend_pred(base)
base_by_w = {w: BA(y, blend_pred(base, w)) for w in W}
print(f"기준선(out/v5, 라벨당 2,500개): n-gram OOF {BA(y, base_ng):.4f} | T_NG {best_t(base)} T_LORA {T_LORA} | "
      f"블렌드 w=0.6 {BA(y, base_bl):.4f} | 최고 w {max(base_by_w, key=base_by_w.get)} 블렌드 {max(base_by_w.values()):.4f}")

order = lambda s: (int(s.split("_")[0][1:]), s)
names = sys.argv[1:] or sorted((Path(f).stem.replace("ngram_oof_", "") for f in glob.glob(str(OUT / "ngram_oof_*.npz"))), key=order)
rows, cost, per_label, boots, preds = [], [], {}, {}, {}
for name in names:
    z = np.load(OUT / f"ngram_oof_{name}.npz", allow_pickle=True)
    ng = z["oof_ng"]
    meta = json.loads((OUT / f"cache_n{int(z['n'])}" / "meta.json").read_text())
    p_ng, p_bl = ng.argmax(1), blend_pred(ng)
    by_w = {w: BA(y, blend_pred(ng, w)) for w in W}
    d_ng, d_bl = boot_ba(p_ng) - boot_ba(base_ng), boot_ba(p_bl) - boot_ba(base_bl)
    boots[name] = (d_ng, d_bl)
    preds[name] = (p_ng, p_bl)
    best_w = max(by_w, key=by_w.get)
    tw = float(z["train_weight"])
    rows.append({
        "실행": name, "라벨당": int(z["n"]), "풀 행": int(z["rows"]), "train 가중치": tw,
        "train 몫": tw * len(y) / (tw * len(y) + int(z["rows"])), "라벨 균형": bool(z["balanced"]), "T_NG": best_t(ng),
        "n-gram OOF": BA(y, p_ng), "n-gram Δ": BA(y, p_ng) - BA(y, base_ng), "n-gram Δ 95%": ci(d_ng),
        "블렌드 w=0.6": BA(y, p_bl), "블렌드 Δ": BA(y, p_bl) - BA(y, base_bl), "블렌드 Δ 95%": ci(d_bl),
        "P(블렌드 Δ>0)": float((d_bl > 0).mean()), "최고 w": best_w, "최고 w 블렌드": by_w[best_w],
        "fold0 블렌드": BA(y[fold == 0], p_bl[fold == 0]), "바뀐 블렌드 예측": int((p_bl != base_bl).sum()),
        "누수 통제 코사인": float(z["leak_tau"]) if "leak_tau" in z and float(z["leak_tau"]) > 0 else np.nan,
        "fold당 뺀 풀 행 비율": float(np.mean(z["dropped"]) / int(z["rows"])) if "dropped" in z else 0.0,
    })
    cost.append({
        "실행": name, "풀 행": int(z["rows"]), "nnz(M)": meta["nnz_pool"] / 1e6, "특성 수": meta["n_features"],
        "벡터화(분)": meta["prepare_min"], "벡터화 최대 메모리(GB)": meta["prepare_peak_anon_gb"],
        "fit 5-fold(분)": float(z["fit_min"]), "fold당(분)": float(np.mean(z["fold_min"])), "fit 최대 메모리(GB)": float(z["peak_anon_gb"]),
        "노트북 방식 예상 메모리(GB)": 1.5 + 53 * (meta["nnz_pool"] + meta["nnz_train"]) / 2**30,  # alpha_sweep 실측(2.2 + 4.7GB)에 맞춘 어림값
        "liblinear 반복": int(np.max(z["n_iter"])),
        "match_weight 평균": meta["match_weight_mean"], "train 대 표본 AUC": meta["drift_auc"],
        "라벨 평균 거리": meta["drift_label_dist_mean"], "거리 최대": meta["drift_label_dist_max"],
    })
    per_label[name] = (recall(p_ng) - recall(base_ng), recall(p_bl) - recall(base_bl))
    print(f"{name} 블렌드 w별: {({w: round(v, 4) for w, v in by_w.items()})}")
    print(f"{name} 기준선 대비 w별 Δ: {({w: round(by_w[w] - base_by_w[w], 4) for w in W})}")

fmt = lambda v: f"{v:.4f}"
print("\n== 요약 (Δ는 기준선 대비, 구간은 프롬프트 paired bootstrap 1,000회) ==")
print(pd.DataFrame(rows).set_index("실행").to_string(float_format=fmt))

print("\n== 추세: 가중치를 건드리지 않은 실행(n<숫자>)만, 표본 크기 순 ==")
plain = [n for n in names if "_" not in n and n != "n2500"]
if plain:
    print("n-gram Δ:", {n: round(float(BA(y, np.load(OUT / f'ngram_oof_{n}.npz')['oof_ng'].argmax(1)) - BA(y, base_ng)), 4) for n in plain})
    print("블렌드 Δ:", {r["실행"]: round(float(r["블렌드 Δ"]), 4) for r in rows if r["실행"] in plain})
for group, title in ((plain, "가중치를 건드리지 않은 실행"), ([n for n in names if n != "n2500"], "2,500개를 넘는 실행 전부")):
    for label, key, i in (("n-gram", "n-gram Δ", 0), ("블렌드 w=0.6", "블렌드 Δ", 1)):
        if group:
            m = np.mean([boots[n][i] for n in group], axis=0)
            point = np.mean([r[key] for r in rows if r["실행"] in group])
            print(f"{title} {len(group)}개의 {label} Δ 평균: {point:+.4f} {ci(m)}, P(평균 Δ>0) {float((m > 0).mean()):.3f}")

print("\n== 실행끼리의 비교 (뒤 - 앞). _lk, _lk10은 검증 fold의 프롬프트와 주제가 가까운 풀 행을 뺀 누수 통제 실행 ==")
pairs = [("n2500", "n2500_lk"), ("n2500", "n2500_lk10"), ("n5000", "n5000_lk"), ("n5000", "n5000_lk10"), ("n7500", "n7500_lk10"),
         ("n2500_lk", "n5000_lk"), ("n2500_lk10", "n5000_lk10"), ("n2500_lk10", "n7500_lk10"), ("n5000_lk10", "n7500_lk10"),
         ("n2500", "n5000"), ("n2500", "n7500"), ("n5000", "n5000_tw"), ("n5000", "n5000_bal"), ("n7500", "n7500_bal"),
         ("n2500", "n5000_bal"), ("n2500", "n7500_bal"), ("n5000_bal", "n7500_bal"), ("n5000", "n7500")]
for a, b in pairs:
    if a in preds and b in preds:
        out = []
        for label, i in (("n-gram", 0), ("블렌드 w=0.6", 1)):
            dd = boot_ba(preds[b][i]) - boot_ba(preds[a][i])
            out.append(f"{label} {BA(y, preds[b][i]) - BA(y, preds[a][i]):+.4f} {ci(dd)}")
        print(f"{b} - {a}: " + " | ".join(out))

print("\n== 비용과 train에서 멀어진 정도 ==")
print(pd.DataFrame(cost).set_index("실행").to_string(float_format=lambda v: f"{v:.3f}"))
seen = set()
for name in names:
    n = int(name.split("_")[0][1:])
    if n in seen:
        continue
    seen.add(n)
    meta = json.loads((OUT / f"cache_n{n}" / "meta.json").read_text())
    print(f"n={n}: 라벨별 행 수 {dict(zip(LABELS, meta['counts']))}")
    print(f"n={n}: 라벨별 match_weight 평균 {dict(zip(LABELS, np.round(meta['match_weight_by_label'], 2).tolist()))}")
    print(f"n={n}: 라벨별로 쓴 비율 {dict(zip(LABELS, np.round(meta['share_used'], 2).tolist()))}")

print("\n== 라벨별 recall 변화: n-gram (기준선 recall, 실행별 Δ) ==")
print(pd.DataFrame({"기준선": recall(base_ng), **{n: per_label[n][0] for n in names}}, index=LABELS).to_string(float_format=lambda v: f"{v:+.3f}"))
print("\n== 라벨별 recall 변화: 블렌드 w=0.6 ==")
print(pd.DataFrame({"기준선": recall(base_bl), **{n: per_label[n][1] for n in names}}, index=LABELS).to_string(float_format=lambda v: f"{v:+.3f}"))
