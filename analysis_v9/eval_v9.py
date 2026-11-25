"""v9 LoRA OOF(Open Pool 전체로 Stage 1)를 v4 LoRA OOF와 비교하고, 3분기 블렌드 가중치를 다시 고릅니다 (CPU, 재학습 없음).

  - v9 LoRA OOF: 첫째 인자(기본 analysis_v9/ntl_lora_v9_oof.npz. Colab 실행이 저장한 파일. 키 oof, test, y, fold, ids)
                 커널이 저장한 oof_v9.npz(키 lora_logits)를 줘도 됩니다.
  - v4 LoRA OOF, n-gram(v8), DeBERTa(v7): analysis_v8/oof_v8.npz (v8 제출 커널이 저장한 OOF 로짓)
  - fold: analysis_v4/oof_v4.npz의 fold (= Kaggle 커널의 분할). v9 npz의 fold와 같은지 확인합니다.
계산은 노트북 5장·5-1장과 같습니다(analysis_v7/whatif_ngram.py와 같은 식): 분기마다 NLL이 가장 작은 온도, 0.1 간격 가중치 표,
fold 하나를 빼고 고르는 nested, 회사 게이트(C = 1, gate_alpha 그대로). 게이트의 LR은 이 서버의 scikit-learn으로 다시 학습하므로
커널 로그와 지문 1개쯤 다를 수 있습니다. 차이의 구간은 프롬프트 150개 단위 paired bootstrap 1,000회입니다.

사용: .venv/bin/python analysis_v9/eval_v9.py [v9 npz] > analysis_v9/logs/eval_v9.log
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
FIXED = dict(lora=0.6, deb=0.2, ng=0.2)  # v7, v8에서 고정한 blend3

v4 = np.load(ROOT / "analysis_v4/oof_v4.npz", allow_pickle=True)
y, fold, pid = v4["y"], v4["fold"], v4["prompt_id"]
assert list(v4["labels"]) == LABELS
folds = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(5)]
v8 = np.load(ROOT / "analysis_v8/oof_v8.npz")
assert (v8["y"] == y).all()
ng, deb, lora4 = v8["ng_logits"], v8["deb_logits"], v8["lora_logits"]
path9 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "analysis_v9/ntl_lora_v9_oof.npz"
z9 = np.load(path9)
assert (z9["y"] == y).all() and (z9["ids"] == v8["ids"]).all(), "v9 OOF의 행 순서가 v8 OOF와 다릅니다."
if "fold" in z9.files:
    assert (z9["fold"] == fold).all(), "v9 OOF의 fold가 Kaggle 커널 분할과 다릅니다."
lora9 = z9["oof"] if "oof" in z9.files else z9["lora_logits"]
print(f"v9 LoRA OOF: {path9} | n = {len(y)}, 프롬프트 {len(np.unique(pid))}개 | fold는 analysis_v4/oof_v4.npz와 같음")


def lsm(x, t=1.0):
    x = x / t
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def best_t(s, grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3)):
    return grid[int(np.argmin([-lsm(s, t)[np.arange(len(y)), y].mean() for t in grid]))]


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
            print(f"  [{name}] 회사 안 정확도 블렌드 {wa(blend_lp):.3f}, 특화 분류기 {wa_spec(spec, cols, inC):.3f} | alpha {GATE_ALPHA[name]} 사용, alpha별 OOF BA {grid}")
        out[:, cols] = apply(GATE_ALPHA[name])
    return out


def wa_spec(spec, cols, inC):
    return (spec[inC].argmax(1) == np.searchsorted(cols, y[inC])).mean()


def evaluate(lora, verbose=False):
    T = {"ng": best_t(ng), "lora": best_t(lora), "deb": best_t(deb)}
    L = {"ng": lsm(ng, T["ng"]), "lora": lsm(lora, T["lora"]), "deb": lsm(deb, T["deb"])}
    blend = lambda w: sum(w[k] * L[k] for k in w if w[k])
    preds = np.stack([blend(w).argmax(1) for w in GRID])
    scores = np.array([ba(p) for p in preds])
    nested, picks = np.zeros(len(y), dtype=int), []
    for a, b in folds:
        i = int(np.argmax([ba(p, a) for p in preds]))
        nested[b] = preds[i][b]
        picks.append(GRID[i])
    best_w = GRID[int(scores.argmax())]
    lps = [L["ng"], L["lora"], L["deb"]]
    # nested 게이트: fold마다 고른 가중치로 블렌드한 뒤 게이트(특화 LR은 원래 fold 밖 학습)
    return dict(T=T, L=L, scores=scores, preds=preds, fixed=blend(FIXED).argmax(1), best_w=best_w, best=scores.max(),
                nested=nested, picks=picks, blend=blend, lps=lps,
                gate_fixed=gate(lps, blend(FIXED), verbose).argmax(1), gate_best=gate(lps, blend(best_w)).argmax(1),
                by_lora={g: float(max(s for s, w in zip(scores, GRID) if w["lora"] == g)) for g in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)})


print("\n== LoRA 단독 (로짓 argmax)")
p4, p9, pd_ = lora4.argmax(1), lora9.argmax(1), deb.argmax(1)
print(f"v4 LoRA (표본 45,000): OOF BA {ba(p4):.4f} | fold별 {per_fold(p4)} | 온도 {best_t(lora4)}")
print(f"v9 LoRA (풀 전체)    : OOF BA {ba(p9):.4f} | fold별 {per_fold(p9)} | 온도 {best_t(lora9)}")
print(f"v9 - v4 (paired bootstrap): {paired(p9, p4)}")
nll = lambda s: -lsm(s, best_t(s))[np.arange(len(y)), y].mean()
print(f"NLL(온도 조정 뒤): v4 {nll(lora4):.4f}, v9 {nll(lora9):.4f}")
print(f"1순위가 다른 지문: v9 vs v4 LoRA {(p9 != p4).mean():.1%} ({(p9 != p4).sum()}개) | v9 LoRA vs DeBERTa {(p9 != pd_).mean():.1%} "
      f"(v4 LoRA vs DeBERTa {(p4 != pd_).mean():.1%})")
print(f"정확도: v4 {(p4 == y).mean():.1%}, v9 {(p9 == y).mean():.1%}, DeBERTa {(pd_ == y).mean():.1%} | "
      f"v9와 DeBERTa 중 하나라도 맞힘 {((p9 == y) | (pd_ == y)).mean():.1%} (v4와 DeBERTa {((p4 == y) | (pd_ == y)).mean():.1%}) | "
      f"v4만 맞힘 {((p4 == y) & (p9 != y)).sum()}개, v9만 맞힘 {((p9 == y) & (p4 != y)).sum()}개")
# Stage 2의 검증 손실: fold 밖 예측(OOF) 로짓의 평균 cross-entropy (온도 1, label smoothing·가중치 없음). fold마다 값 하나이고 곡선은 없습니다.
ce = lambda s, idx=slice(None): float(-lsm(s)[np.arange(len(y)), y][idx].mean())
print("Stage 2 검증 손실(OOF 평균 cross-entropy, 온도 1): "
      f"v9 전체 {ce(lora9):.4f} | fold별 " + " / ".join(f"{ce(lora9, b):.4f}" for _, b in folds)
      + f" || v4 전체 {ce(lora4):.4f} | fold별 " + " / ".join(f"{ce(lora4, b):.4f}" for _, b in folds))
print("라벨별 recall (v4 -> v9):")
for k, name in enumerate(LABELS):
    m = y == k
    print(f"  {name:<24} {(p4[m] == k).mean():.3f} -> {(p9[m] == k).mean():.3f} ({(p9[m] == k).mean() - (p4[m] == k).mean():+.3f})")

res = {}
for name, lora in (("v8 제출 (v4 LoRA)", lora4), ("v9 (풀 전체 LoRA)", lora9)):
    print(f"\n== 3분기 블렌드: {name}  [n-gram v8 {ba(ng.argmax(1)):.4f}, DeBERTa v7 {ba(deb.argmax(1)):.4f}]")
    r = res[name] = evaluate(lora, verbose=True)
    print(f"온도: {r['T']}")
    print(f"고정 가중치({fmt(FIXED)}) {ba(r['fixed']):.4f} (fold별 {per_fold(r['fixed'])}) | OOF 최고 {r['best']:.4f} ({fmt(r['best_w'])}) | nested {ba(r['nested']):.4f}")
    for i in np.argsort(-r["scores"])[:8]:
        print(f"  OOF BA {r['scores'][i]:.4f} | {fmt(GRID[i])}")
    print("nested가 fold마다 고른 가중치:", "; ".join(fmt(w) for w in r["picks"]))
    print("LoRA 가중치별 최고 OOF BA(나머지 두 가중치는 자유):", {k: round(v, 4) for k, v in r["by_lora"].items()})
    print(f"게이트(gate_alpha 그대로): 고정 가중치 {ba(r['fixed']):.4f} -> {ba(r['gate_fixed']):.4f} (바뀐 지문 {(r['fixed'] != r['gate_fixed']).sum()}개) | "
          f"OOF 최고 가중치 {r['best']:.4f} -> {ba(r['gate_best']):.4f}")

a, b = res["v8 제출 (v4 LoRA)"], res["v9 (풀 전체 LoRA)"]
print("\n== v9 - v8 제출(v4 LoRA, 고정 가중치) paired bootstrap: 차이 [95% 구간] P(차이>0)")
print(f"  블렌드: v9 고정 가중치 - v8 고정      : {paired(b['fixed'], a['fixed'])}")
print(f"  블렌드: v9 nested - v8 고정           : {paired(b['nested'], a['fixed'])}")
print(f"  블렌드: v9 nested - v8 nested         : {paired(b['nested'], a['nested'])}")
print(f"  블렌드: v9 OOF 최고 - v8 고정         : {paired(b['preds'][int(b['scores'].argmax())], a['fixed'])}")
print(f"  게이트 뒤: v9 고정 - v8 고정          : {ba(a['gate_fixed']):.4f} -> {ba(b['gate_fixed']):.4f} | {paired(b['gate_fixed'], a['gate_fixed'])}")
print(f"  게이트 뒤: v9 OOF 최고 - v8 고정      : {ba(a['gate_fixed']):.4f} -> {ba(b['gate_best']):.4f} | {paired(b['gate_best'], a['gate_fixed'])}")
print(f"  v9 안에서: OOF 최고 - 고정 (블렌드)   : {paired(b['preds'][int(b['scores'].argmax())], b['fixed'])}")
print(f"  v9 안에서: nested - 고정 (블렌드)     : {paired(b['nested'], b['fixed'])}")
print(f"  v9 안에서: OOF 최고 - 고정 (게이트 뒤): {paired(b['gate_best'], b['gate_fixed'])}")

print("\n== 블렌드(고정 가중치)에서 바뀐 지문과 라벨별 recall: v8 제출 -> v9")
print(f"블렌드 예측이 바뀐 지문 {(a['fixed'] != b['fixed']).sum()}개 | 고침 {((a['fixed'] != y) & (b['fixed'] == y)).sum()}개, 망침 {((a['fixed'] == y) & (b['fixed'] != y)).sum()}개 | "
      f"게이트 뒤: 바뀜 {(a['gate_fixed'] != b['gate_fixed']).sum()}개, 고침 {((a['gate_fixed'] != y) & (b['gate_fixed'] == y)).sum()}개, 망침 {((a['gate_fixed'] == y) & (b['gate_fixed'] != y)).sum()}개")
print(f"오답 중 같은 연구소 모델로 틀린 비율(게이트 뒤): " + ", ".join(
    f"{n} {np.mean([any(p in FAMILIES[f] and t in FAMILIES[f] for f in FAMILIES) for p, t in zip(r['gate_fixed'][r['gate_fixed'] != y], y[r['gate_fixed'] != y])]):.1%}"
    for n, r in (("v8", a), ("v9", b))))
print("라벨: LoRA 단독 | 3분기 블렌드 | 게이트 뒤")
for k, name in enumerate(LABELS):
    m = y == k
    rc = lambda p: (p[m] == k).mean()
    print(f"  {name:<24} {rc(p4):.3f} -> {rc(p9):.3f} | {rc(a['fixed']):.3f} -> {rc(b['fixed']):.3f} | {rc(a['gate_fixed']):.3f} -> {rc(b['gate_fixed']):.3f} ({rc(b['gate_fixed']) - rc(a['gate_fixed']):+.3f})")
print(f"2분기(DeBERTa 없음) lora 0.6 / ng 0.4: v8 {ba(a['blend'](dict(lora=0.6, ng=0.4)).argmax(1)):.4f} -> v9 {ba(b['blend'](dict(lora=0.6, ng=0.4)).argmax(1)):.4f} | "
      f"v9에서 DeBERTa의 몫(고정 3분기 - 2분기): {paired(b['fixed'], b['blend'](dict(lora=0.6, ng=0.4)).argmax(1))}")
print(f"v9에서 n-gram의 몫(고정 3분기 - lora 0.8 / deb 0.2): {paired(b['fixed'], b['blend'](dict(lora=0.8, deb=0.2)).argmax(1))}")
