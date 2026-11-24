"""n-gram OOF만 바꿔 가며 v7의 3분기 블렌드와 회사 게이트를 다시 계산합니다 (CPU, 재학습·제출 없음).

질문: n-gram 풀 표본을 라벨당 2,500개에서 5,000개로 늘려 얻은 이득(analysis_pool/)이 DeBERTa가 들어간 블렌드에서도
남는가, 그때 블렌드는 n-gram 가중치를 얼마로 원하는가.

  - LoRA OOF   : analysis_v4/oof_v4.npz 의 lora_oof (커널 fold, 0.5631)
  - DeBERTa OOF: out/v7/colab/ntl_deb_oof.npz (0.5053). 없으면 out/v7/kernel/oof_v7.npz 의 deb_logits (같은 값)
  - n-gram OOF : 인자로 준 npz들의 oof_ng. 첫 번째가 비교 기준(제출한 v7과 같은 설정을 로컬에서 잰 것),
                 두 번째가 "풀 크기만의 효과"의 기준(문체 블록 포함, 풀 2,500). 마지막 줄은 제출 커널이 저장한 OOF(analysis_v7/oof_v7.npz)
계산은 노트북 5장·5-1장과 같습니다: 분기마다 NLL이 가장 작은 온도, 0.1 간격 가중치 표, fold 하나를 빼고 고르는 nested,
회사 게이트(C = 1, gate_alpha는 v4 OOF에서 고른 값 그대로). fold는 oof_v4.npz의 fold(= Kaggle 커널의 분할)입니다.
차이의 구간은 프롬프트 150개 단위 paired bootstrap 1,000회입니다.

사용: .venv/bin/python analysis_v7/whatif_ngram.py  [이름=npz 경로 ...]
"""
import itertools, os, sys

import numpy as np
from scipy.special import log_softmax, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bas

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
DEFAULT = [
    ("v7 제출 설정 (풀 2,500, 문체 없음)", "v5_stylo/ngram_oof_alpha0_kernelfold.npz"),
    ("+ 문체 α=0.2 (풀 2,500)", "v5_stylo/ngram_oof_alpha0.2_kernelfold.npz"),
    ("+ 문체, 풀 5,000", "out/pool/ngram_oof_n5000.npz"),
    ("+ 문체, 풀 5,000, 라벨 균형", "out/pool/ngram_oof_n5000_bal.npz"),
    ("+ 문체, 풀 7,500", "out/pool/ngram_oof_n7500.npz"),
    ("+ 문체, 풀 7,500, 라벨 균형", "out/pool/ngram_oof_n7500_bal.npz"),
    ("참고: 제출 커널이 저장한 n-gram OOF", "analysis_v7/oof_v7.npz"),
]
runs = [tuple(a.split("=", 1)) for a in sys.argv[1:]] or DEFAULT


def load_ng(path):
    z = np.load(P(path))
    return z["oof_ng"] if "oof_ng" in z.files else z["ng_logits"]  # 커널의 oof_v7.npz는 키 이름이 다릅니다.

LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
          "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
          "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
          "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]
FAMILIES = {p: [i for i, l in enumerate(LABELS) if l.startswith(p)] for p in ("gpt", "gemini", "claude", "glm")}
GATE_ALPHA, GATE_C = dict(gpt=0.75, gemini=0.25, claude=0.5, glm=1.0), 1.0
FIXED = dict(lora=0.6, deb=0.2, ng=0.2)       # v7에서 고정한 blend3
V6W = dict(lora=0.6, deb=0.0, ng=0.4)

v4 = np.load(P("analysis_v4/oof_v4.npz"), allow_pickle=True)
y, fold, pid = v4["y"], v4["fold"], v4["prompt_id"]
assert list(v4["labels"]) == LABELS
folds = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(5)]
deb_path = P("out/v7/colab/ntl_deb_oof.npz")
deb = np.load(deb_path)["oof"] if os.path.exists(deb_path) else np.load(P("out/v7/kernel/oof_v7.npz"))["deb_logits"]


def lsm(x, t=1.0):
    x = x / t
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def best_t(s, grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3)):
    return grid[int(np.argmin([-lsm(s, t)[np.arange(len(y)), y].mean() for t in grid]))]


ba = lambda pred, idx=slice(None): bas(y[idx], pred[idx])
GRID = [dict(lora=a / 10, deb=b / 10, ng=round(1 - a / 10 - b / 10, 1))
        for a, b in itertools.product(range(11), repeat=2) if a + b <= 10]
fmt = lambda w: f"lora {w['lora']:.1f} / deb {w['deb']:.1f} / ng {w['ng']:.1f}"


def gate(lps, blend_lp):
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
        new = log_softmax((1 - GATE_ALPHA[name]) * (sub - mass) + GATE_ALPHA[name] * spec, axis=1) + mass
        out[:, cols] = np.where(np.isin(out.argmax(1), cols)[:, None], new, sub)
    return out


def evaluate(oof_ng):
    L = {"ng": lsm(oof_ng, best_t(oof_ng)), "lora": lsm(v4["lora_oof"], best_t(v4["lora_oof"])), "deb": lsm(deb, best_t(deb))}
    blend = lambda w: sum(w[k] * L[k] for k in w if w[k])
    preds = np.stack([blend(w).argmax(1) for w in GRID])
    scores = np.array([ba(p) for p in preds])
    nested, picks = np.zeros(len(y), dtype=int), []
    for a, b in folds:
        i = int(np.argmax([ba(p, a) for p in preds]))
        nested[b] = preds[i][b]
        picks.append(GRID[i])
    two = [(ba(blend(dict(lora=w / 10, deb=0.0, ng=round(1 - w / 10, 1))).argmax(1)), w / 10) for w in range(11)]
    fixed_lp, best_w = blend(FIXED), GRID[int(scores.argmax())]
    lps = [L["ng"], L["lora"], L["deb"]]
    return dict(T=best_t(oof_ng), ng=ba(oof_ng.argmax(1)), fixed=fixed_lp.argmax(1), best_w=best_w, best=scores.max(),
                nested=nested, picks=picks, v6w=ba(blend(V6W).argmax(1)), two=max(two),
                gate_fixed=gate(lps, fixed_lp).argmax(1), gate_best=gate(lps, blend(best_w)).argmax(1),
                v6_pred=blend(V6W).argmax(1), v6_gate=gate(lps[:2], blend(V6W)).argmax(1),  # v6: 2분기 블렌드, 게이트 입력도 2분기
                by_ng={g: float(max(s for s, w in zip(scores, GRID) if w["ng"] == g)) for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)})


prompts = np.unique(pid)
rows_of = {p: np.where(pid == p)[0] for p in prompts}
rng = np.random.default_rng(0)
BOOT = [np.concatenate([rows_of[p] for p in rng.choice(prompts, len(prompts))]) for _ in range(1000)]


def paired(pred_b, pred_a):
    d = np.array([bas(y[i], pred_b[i]) - bas(y[i], pred_a[i]) for i in BOOT])
    return f"{ba(pred_b) - ba(pred_a):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] P>0 {np.mean(d > 0):.2f}"


print(f"LoRA OOF BA {ba(v4['lora_oof'].argmax(1)):.4f} (T {best_t(v4['lora_oof'])}) | DeBERTa OOF BA {ba(deb.argmax(1)):.4f} (T {best_t(deb)}) | n = {len(y)}, 프롬프트 {len(prompts)}개")
res = {}
for name, path in runs:
    r = res[name] = evaluate(load_ng(path))
    print(f"\n== {name}  ({path})")
    print(f"n-gram OOF BA {r['ng']:.4f} (T {r['T']}) | 2분기 v6 가중치(lora 0.6 / ng 0.4) {r['v6w']:.4f}, 2분기 최고 {r['two'][0]:.4f} (lora {r['two'][1]})")
    print(f"3분기 고정 가중치({fmt(FIXED)}) {ba(r['fixed']):.4f} | OOF 최고 {r['best']:.4f} ({fmt(r['best_w'])}) | nested {ba(r['nested']):.4f}")
    print("nested가 fold마다 고른 가중치:", "; ".join(fmt(w) for w in r["picks"]))
    print("n-gram 가중치별 최고 OOF BA(나머지 두 가중치는 자유):", {k: round(v, 4) for k, v in r["by_ng"].items()})
    print(f"게이트(gate_alpha 그대로): 고정 가중치 {ba(r['fixed']):.4f} -> {ba(r['gate_fixed']):.4f} | OOF 최고 가중치 {r['best']:.4f} -> {ba(r['gate_best']):.4f}")

print("\n== DeBERTa의 효과: 같은 n-gram OOF에서 v6(lora 0.6 / ng 0.4, 게이트 입력 2분기) 대비 paired bootstrap")
for name, _ in runs:
    r = res[name]
    print(f"{name}")
    print(f"  3분기 고정 - v6 블렌드         : {paired(r['fixed'], r['v6_pred'])}")
    print(f"  3분기 nested - v6 블렌드       : {paired(r['nested'], r['v6_pred'])}")
    print(f"  게이트 뒤: 3분기 고정 - v6     : {ba(r['v6_gate']):.4f} -> {ba(r['gate_fixed']):.4f} | {paired(r['gate_fixed'], r['v6_gate'])}")

base = res[runs[0][0]]
print(f"\n== 기준({runs[0][0]}) 대비 paired bootstrap: 차이 [95% 구간] P(차이>0)")
for name, _ in runs[1:]:
    r = res[name]
    print(f"{name}")
    print(f"  n-gram 단독            : {paired(load_ng(dict(runs)[name]).argmax(1), load_ng(runs[0][1]).argmax(1))}")
    print(f"  3분기 고정 가중치      : {paired(r['fixed'], base['fixed'])}")
    print(f"  3분기 nested           : {paired(r['nested'], base['nested'])}")
    print(f"  nested vs 기준 고정    : {paired(r['nested'], base['fixed'])}")
    print(f"  고정 가중치 + 게이트   : {paired(r['gate_fixed'], base['gate_fixed'])}")

if len(runs) > 2:
    ref_name, ref = runs[1][0], res[runs[1][0]]
    print(f"\n== 풀 크기만의 효과: 기준({ref_name}) 대비 paired bootstrap (2분기는 lora 0.6 / ng 0.4)")
    L_lora = lsm(v4["lora_oof"], best_t(v4["lora_oof"]))
    two = lambda path: (0.6 * L_lora + 0.4 * lsm(load_ng(path), best_t(load_ng(path)))).argmax(1)
    for name, path in runs[2:]:
        if "참고" in name:
            continue
        r = res[name]
        print(f"{name}")
        print(f"  2분기 (DeBERTa 없음)   : {paired(two(path), two(runs[1][1]))}")
        print(f"  3분기 고정 가중치      : {paired(r['fixed'], ref['fixed'])}")
        print(f"  3분기 nested           : {paired(r['nested'], ref['nested'])}")
        print(f"  고정 가중치 + 게이트   : {paired(r['gate_fixed'], ref['gate_fixed'])}")

print("\n== 표 (BA)")
print("| n-gram OOF 설정 | n-gram | 2분기 lora 0.6 / ng 0.4 | 3분기 고정 0.6/0.2/0.2 | OOF 최고 (lora/deb/ng) | nested | 고정 + 게이트 | 최고 + 게이트 |")
print("|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for name, _ in runs:
    r = res[name]
    print(f"| {name} | {r['ng']:.4f} | {r['v6w']:.4f} | {ba(r['fixed']):.4f} | {r['best']:.4f} ({r['best_w']['lora']:.1f}/{r['best_w']['deb']:.1f}/{r['best_w']['ng']:.1f}) | "
          f"{ba(r['nested']):.4f} | {ba(r['gate_fixed']):.4f} | {ba(r['gate_best']):.4f} |")
