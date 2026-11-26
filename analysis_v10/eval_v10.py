"""v10 DeBERTa OOF(Open Pool 전체로 Stage 1)를 v7 DeBERTa OOF와 비교하고, 3분기 블렌드 -> 회사 게이트 -> Sinkhorn을 다시 계산합니다 (CPU, 재학습 없음).

  - v10 DeBERTa OOF: 첫째 인자(기본 out/v10/ntl_deb_v10_oof.npz. Colab 실행이 저장한 파일. 키 oof, ids, y, fold)
                     제출 커널이 저장한 oof_v10.npz(키 deb_logits)를 줘도 됩니다.
  - n-gram(v8), LoRA(v9), v7 DeBERTa: analysis_v9/oof_v9.npz (v9 제출 커널이 저장한 OOF 로짓)
  - fold: analysis_v4/oof_v4.npz의 fold (= Kaggle 커널의 분할). v10 npz의 fold와 같은지 확인합니다.
계산은 노트북 5장·5-1장·7장과 같습니다(analysis_v9/eval_v9.py와 같은 식): 분기마다 NLL이 가장 작은 온도, 0.1 간격 가중치 표,
fold 하나를 빼고 고르는 nested, 회사 게이트(C = 1, gate_alpha 그대로), OOF 전체에 Sinkhorn(rho 0.35). 게이트의 LR은 이 서버의
scikit-learn으로 다시 학습하므로 커널 로그와 지문 1개쯤 다를 수 있습니다. 차이의 구간은 프롬프트 150개 단위 paired bootstrap 1,000회입니다.

사용: .venv/bin/python analysis_v10/eval_v10.py [v10 npz] > analysis_v10/logs/eval_v10.log
"""
import itertools, sys
from pathlib import Path

import numpy as np
from scipy.special import log_softmax, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bas

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
          "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
          "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
          "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]
FAMILIES = {p: [i for i, l in enumerate(LABELS) if l.startswith(p)] for p in ("gpt", "gemini", "claude", "glm")}
GATE_ALPHA, GATE_C = dict(gpt=0.75, gemini=0.25, claude=0.5, glm=1.0), 1.0
FIXED = dict(lora=0.6, deb=0.2, ng=0.2)  # v7부터 고정한 blend3

v4 = np.load(ROOT / "analysis_v4/oof_v4.npz", allow_pickle=True)
y, fold, pid = v4["y"], v4["fold"], v4["prompt_id"]
assert list(v4["labels"]) == LABELS
folds = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(5)]
v9 = np.load(ROOT / "analysis_v9/oof_v9.npz")
assert (v9["y"] == y).all()
ng, lora, deb7 = v9["ng_logits"], v9["lora_logits"], v9["deb_logits"]
path10 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out/v10/ntl_deb_v10_oof.npz"
z = np.load(path10)
assert (z["ids"] == v9["ids"]).all(), "v10 OOF의 행 순서가 v9 OOF와 다릅니다."
if "y" in z.files:
    assert (z["y"] == y).all()
if "fold" in z.files:
    assert (z["fold"] == fold).all(), "v10 OOF의 fold가 Kaggle 커널 분할과 다릅니다."
deb10 = z["oof"] if "oof" in z.files else z["deb_logits"]
print(f"v10 DeBERTa OOF: {path10} | n = {len(y)}, 프롬프트 {len(np.unique(pid))}개 | fold는 analysis_v4/oof_v4.npz와 같음")


def lsm(x, t=1.0):
    x = x / t
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def best_t(s, grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3)):
    return grid[int(np.argmin([-lsm(s, t)[np.arange(len(y)), y].mean() for t in grid]))]


def sinkhorn_balance(lp, rho=0.35, n_iter=10, damping=0.5):
    def softmax(v):
        v = v - v.max(1, keepdims=True)
        e = np.exp(v)
        return e / e.sum(1, keepdims=True)
    target = (1 - rho) * softmax(lp).mean(0) + rho / lp.shape[1]
    bias = np.zeros(lp.shape[1])
    for _ in range(n_iter):
        bias += damping * (np.log(target) - np.log(softmax(lp + bias).mean(0)))
    return lp + bias


ba = lambda pred, idx=slice(None): bas(y[idx], pred[idx])
GRID = [dict(lora=a / 10, deb=b / 10, ng=round(1 - a / 10 - b / 10, 1))
        for a, b in itertools.product(range(11), repeat=2) if a + b <= 10]
fmt = lambda w: f"lora {w['lora']:.1f} / deb {w['deb']:.1f} / ng {abs(w['ng']):.1f}"
per_fold = lambda pred: " / ".join(f"{ba(pred, b):.4f}" for _, b in folds)

prompts = np.unique(pid)
rows_of = {p: np.where(pid == p)[0] for p in prompts}
rng = np.random.default_rng(0)
BOOT = [np.concatenate([rows_of[p] for p in rng.choice(prompts, len(prompts))]) for _ in range(1000)]


def paired(pred_b, pred_a):
    d = np.array([bas(y[i], pred_b[i]) - bas(y[i], pred_a[i]) for i in BOOT])
    return f"{ba(pred_b) - ba(pred_a):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] P>0 {np.mean(d > 0):.2f}"


def ci(pred):
    d = np.array([bas(y[i], pred[i]) for i in BOOT])
    return f"{ba(pred):.4f} [{np.percentile(d, 2.5):.4f}, {np.percentile(d, 97.5):.4f}]"


def gate(lps, blend_lp, verbose=False):
    """노트북 5-1장과 같은 회사 게이트. 게이트 뒤 log 확률을 돌려줍니다."""
    out = blend_lp.copy()
    for name, cols in FAMILIES.items():
        feat = np.hstack([lp[:, cols] for lp in lps])
        spec = np.zeros((len(y), len(cols)))
        for a, b in folds:
            rows = a[np.isin(y[a], cols)]
            m = LogisticRegression(C=GATE_C, max_iter=3000).fit(feat[rows], np.searchsorted(cols, y[rows]))
            spec[b] = m.predict_log_proba(feat[b])
        sub = out[:, cols]
        mass = logsumexp(sub, axis=1, keepdims=True)
        apply = lambda al: np.where(np.isin(out.argmax(1), cols)[:, None],
                                    log_softmax((1 - al) * (sub - mass) + al * spec, axis=1) + mass, sub)
        if verbose:
            grid = {}
            for al in (0, 0.25, 0.5, 0.75, 1.0):
                tmp = out.copy()
                tmp[:, cols] = apply(al)
                grid[al] = round(float(ba(tmp.argmax(1))), 4)
            inC = np.isin(y, cols)
            wa = lambda lp: (lp[inC][:, cols].argmax(1) == np.searchsorted(cols, y[inC])).mean()
            ws = (spec[inC].argmax(1) == np.searchsorted(cols, y[inC])).mean()
            print(f"  [{name}] 회사 안 정확도 블렌드 {wa(blend_lp):.3f}, 특화 분류기 {ws:.3f} | alpha {GATE_ALPHA[name]} 사용, alpha별 OOF BA {grid}")
        out[:, cols] = apply(GATE_ALPHA[name])
    return out


def evaluate(deb, verbose=False):
    T = {"ng": best_t(ng), "lora": best_t(lora), "deb": best_t(deb)}
    L = {"ng": lsm(ng, T["ng"]), "lora": lsm(lora, T["lora"]), "deb": lsm(deb, T["deb"])}
    blend = lambda w: sum(w[k] * L[k] for k in w if w[k])
    preds = np.stack([blend(w).argmax(1) for w in GRID])
    scores = np.array([ba(p) for p in preds])
    nested, picks, nested_lp = np.zeros(len(y), dtype=int), [], np.zeros((len(y), len(LABELS)))
    for a, b in folds:
        i = int(np.argmax([ba(p, a) for p in preds]))
        nested[b] = preds[i][b]
        nested_lp[b] = blend(GRID[i])[b]
        picks.append(GRID[i])
    best_w = GRID[int(scores.argmax())]
    lps = [L["ng"], L["lora"], L["deb"]]
    g_fixed = gate(lps, blend(FIXED), verbose)
    g_best = gate(lps, blend(best_w))
    g_nested = gate(lps, nested_lp)
    return dict(T=T, L=L, scores=scores, preds=preds, fixed=blend(FIXED).argmax(1), best_w=best_w, best=scores.max(),
                nested=nested, picks=picks, blend=blend,
                gate_fixed=g_fixed.argmax(1), gate_best=g_best.argmax(1), gate_nested=g_nested.argmax(1),
                sk_fixed=sinkhorn_balance(g_fixed).argmax(1), sk_best=sinkhorn_balance(g_best).argmax(1),
                sk_nested=sinkhorn_balance(g_nested).argmax(1),
                by_deb={g: float(max(s for s, w in zip(scores, GRID) if w["deb"] == g)) for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)})


print("\n== DeBERTa 단독 (로짓 argmax)")
p7, p10, pl = deb7.argmax(1), deb10.argmax(1), lora.argmax(1)
print(f"v7 DeBERTa (표본 45,000, fp32): OOF BA {ci(p7)} | fold별 {per_fold(p7)} | 온도 {best_t(deb7)}")
print(f"v10 DeBERTa (풀 전체, fp16)  : OOF BA {ci(p10)} | fold별 {per_fold(p10)} | 온도 {best_t(deb10)}")
print(f"v10 - v7 (paired bootstrap): {paired(p10, p7)}")
nll = lambda s: -lsm(s, best_t(s))[np.arange(len(y)), y].mean()
ce = lambda s, idx=slice(None): float(-lsm(s)[np.arange(len(y)), y][idx].mean())
print(f"NLL(온도 조정 뒤): v7 {nll(deb7):.4f}, v10 {nll(deb10):.4f} | Stage 2 검증 손실(OOF CE, 온도 1): v7 {ce(deb7):.4f}, v10 {ce(deb10):.4f} "
      f"(fold별 " + " / ".join(f"{ce(deb10, b):.4f}" for _, b in folds) + ")")
print(f"1순위가 다른 지문: v10 vs v7 DeBERTa {(p10 != p7).mean():.1%} | LoRA(v9) vs DeBERTa: v7 {(pl != p7).mean():.1%}, v10 {(pl != p10).mean():.1%}")
print(f"정확도: LoRA {(pl == y).mean():.1%}, DeBERTa v7 {(p7 == y).mean():.1%}, v10 {(p10 == y).mean():.1%} | "
      f"LoRA와 DeBERTa 중 하나라도 맞힘: v7 {((pl == y) | (p7 == y)).mean():.1%}, v10 {((pl == y) | (p10 == y)).mean():.1%}")
print("라벨별 recall (DeBERTa v7 -> v10):")
for k, name in enumerate(LABELS):
    m = y == k
    print(f"  {name:<24} {(p7[m] == k).mean():.3f} -> {(p10[m] == k).mean():.3f} ({(p10[m] == k).mean() - (p7[m] == k).mean():+.3f})")

res = {}
for name, deb in (("v9 제출 (v7 DeBERTa)", deb7), ("v10 (풀 전체 DeBERTa)", deb10)):
    print(f"\n== 3분기 블렌드: {name}  [n-gram v8 {ba(ng.argmax(1)):.4f}, LoRA v9 {ba(lora.argmax(1)):.4f}, DeBERTa {ba(deb.argmax(1)):.4f}]")
    r = res[name] = evaluate(deb, verbose=True)
    print(f"온도: {r['T']}")
    print(f"고정 가중치({fmt(FIXED)}) {ci(r['fixed'])} (fold별 {per_fold(r['fixed'])}) | OOF 최고 {r['best']:.4f} ({fmt(r['best_w'])}) | nested {ba(r['nested']):.4f}")
    for i in np.argsort(-r["scores"])[:8]:
        print(f"  OOF BA {r['scores'][i]:.4f} | {fmt(GRID[i])}")
    print("nested가 fold마다 고른 가중치:", "; ".join(fmt(w) for w in r["picks"]))
    print("DeBERTa 가중치별 최고 OOF BA(나머지 두 가중치는 자유):", {k: round(v, 4) for k, v in r["by_deb"].items()})
    print(f"고정 가중치: 블렌드 {ba(r['fixed']):.4f} -> 게이트 {ba(r['gate_fixed']):.4f} -> 게이트 + Sinkhorn {ba(r['sk_fixed']):.4f} "
          f"(게이트가 바꾼 지문 {(r['fixed'] != r['gate_fixed']).sum()}개)")
    print(f"OOF 최고 가중치: 블렌드 {r['best']:.4f} -> 게이트 {ba(r['gate_best']):.4f} -> 게이트 + Sinkhorn {ba(r['sk_best']):.4f}")
    print(f"nested 가중치: 블렌드 {ba(r['nested']):.4f} -> 게이트 {ba(r['gate_nested']):.4f} -> 게이트 + Sinkhorn {ba(r['sk_nested']):.4f}")

a, b = res["v9 제출 (v7 DeBERTa)"], res["v10 (풀 전체 DeBERTa)"]
print("\n== v10 - v9 제출(v7 DeBERTa) paired bootstrap: 차이 [95% 구간] P(차이>0)")
print(f"  블렌드: v10 고정 - v9 고정               : {paired(b['fixed'], a['fixed'])}")
print(f"  블렌드: v10 nested - v9 고정             : {paired(b['nested'], a['fixed'])}")
print(f"  블렌드: v10 nested - v9 nested           : {paired(b['nested'], a['nested'])}")
print(f"  게이트 뒤: v10 고정 - v9 고정            : {ba(a['gate_fixed']):.4f} -> {ba(b['gate_fixed']):.4f} | {paired(b['gate_fixed'], a['gate_fixed'])}")
print(f"  게이트+Sinkhorn: v10 고정 - v9 고정      : {ba(a['sk_fixed']):.4f} -> {ba(b['sk_fixed']):.4f} | {paired(b['sk_fixed'], a['sk_fixed'])}")
print(f"  v10 안에서: OOF 최고 - 고정 (블렌드)     : {paired(b['preds'][int(b['scores'].argmax())], b['fixed'])}")
print(f"  v10 안에서: nested - 고정 (블렌드)       : {paired(b['nested'], b['fixed'])}")
print(f"  v10 안에서: OOF 최고 - 고정 (게이트 뒤)  : {paired(b['gate_best'], b['gate_fixed'])}")
print(f"  v10 안에서: nested - 고정 (게이트 뒤)    : {paired(b['gate_nested'], b['gate_fixed'])}")
print(f"  v10 안에서: nested - 고정 (게이트+Sinkhorn): {paired(b['sk_nested'], b['sk_fixed'])}")
two = lambda r: r["blend"](dict(lora=0.6, ng=0.4)).argmax(1)
print(f"\n2분기(DeBERTa 없음) lora 0.6 / ng 0.4: {ba(two(b)):.4f} | DeBERTa의 몫(고정 3분기 - 2분기): v9 {paired(a['fixed'], two(a))} | v10 {paired(b['fixed'], two(b))}")
print(f"블렌드 예측이 바뀐 지문 {(a['fixed'] != b['fixed']).sum()}개 | 고침 {((a['fixed'] != y) & (b['fixed'] == y)).sum()}개, 망침 {((a['fixed'] == y) & (b['fixed'] != y)).sum()}개")
print("라벨: DeBERTa 단독 | 3분기 블렌드 | 게이트 뒤 (v9 -> v10)")
for k, name in enumerate(LABELS):
    m = y == k
    rc = lambda p: (p[m] == k).mean()
    print(f"  {name:<24} {rc(p7):.3f} -> {rc(p10):.3f} | {rc(a['fixed']):.3f} -> {rc(b['fixed']):.3f} | {rc(a['gate_fixed']):.3f} -> {rc(b['gate_fixed']):.3f} ({rc(b['gate_fixed']) - rc(a['gate_fixed']):+.3f})")
