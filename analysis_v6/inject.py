"""특화 분류기를 v4 OOF에 넣는 방식 비교 (GPU 없이, train 텍스트 + oof_v4.npz).

특화 분류기 출처 2종 x 넣는 방식 3종 x 묶음 2종.
- 출처 stack: 묶음 열의 n-gram/LoRA OOF log 확률 -> 묶음 안 LR
- 출처 text : 묶음 모델의 train 지문(다른 fold)으로 학습한 TF-IDF LR (train만, 풀 없음)
- 방식 gate : argmax가 묶음 안일 때만, 묶음 확률 질량은 두고 내부 분포를 특화 분류기로 교체(알파 블렌드)
- 방식 v40a: 모든 행에서 묶음 내부에 alpha*잔차를 더하고 묶음 질량(logsumexp)을 원래대로 맞춤
- 방식 free: alpha*잔차를 더하되 질량 보존 없음
alpha는 fold 하나를 빼고 나머지 4 fold BA로 고르는 nested 선택값을 함께 보고한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, log_softmax
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(sys.argv[1])
d = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
L = [str(x) for x in d["labels"]]
I = {l: i for i, l in enumerate(L)}
y, fold, base = d["y"], d["fold"], d["eval_lp"]
ng, lo = d["ng_oof_lp"], d["lora_lp"]
t = pd.read_csv(ROOT / "data" / "train.csv")
key = pd.DataFrame({"prompt_id": d["prompt_id"], "model": [L[i] for i in y]})
text = key.merge(t, on=["prompt_id", "model"], how="left").text.values
N, K = base.shape


def ba(lp):
    p = lp.argmax(1)
    return np.mean([(p[y == k] == k).mean() for k in range(K)])


GROUPS = {
    "company": [[l for l in L if l.startswith(p)] for p in ["gpt", "gemini", "claude", "glm"]],
    "confusion": [
        ["claude-fable-5", "glm-5.3", "glm-5.3-flash"],
        ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"],
        ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
        ["claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5"],
    ],
}

_vec_cache = {}


def text_feats(tr, te):
    v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200000)
    v2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False)
    a = sparse.hstack([v1.fit_transform(text[tr]), v2.fit_transform(text[tr])]).tocsr()
    b = sparse.hstack([v1.transform(text[te]), v2.transform(text[te])]).tocsr()
    return a, b


def specialist(cols, src):
    """묶음 cols에 대한 fold 밖 log 확률 (N, len(cols)). 모든 행에 대해 만든다."""
    out = np.zeros((N, len(cols)))
    inC = np.isin(y, cols)
    yc = np.searchsorted(cols, y) if False else np.array([cols.index(v) if v in cols else -1 for v in y])
    for k in range(5):
        tr = np.where((fold != k) & inC)[0]
        te = np.where(fold == k)[0]
        if src == "stack":
            X = np.hstack([ng[:, cols], lo[:, cols]])
            m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], yc[tr])
            out[te] = m.predict_log_proba(X[te])
        else:
            a, b = text_feats(tr, te)
            m = LogisticRegression(C=10.0, max_iter=3000).fit(a, yc[tr])
            out[te] = m.predict_log_proba(b)
    return out


def apply(lp, cols, S, mode, a):
    lp = lp.copy()
    sub = lp[:, cols]
    mass = logsumexp(sub, 1, keepdims=True)
    inner = sub - mass                               # 묶음 내부 log 분포
    R = S - S.mean(1, keepdims=True)                 # 잔차(중심화)
    if mode == "gate":
        g = np.isin(lp.argmax(1), cols)[:, None]
        new = log_softmax((1 - min(a, 1)) * inner + min(a, 1) * S, axis=1) + mass
        sub = np.where(g, new, sub)
    elif mode == "v40a":
        tl = sub + a * R
        sub = tl - logsumexp(tl, 1, keepdims=True) + mass
    else:  # free
        sub = sub + a * R
    lp[:, cols] = sub
    return lp


ALPHAS = {"gate": [0, .25, .5, .75, 1.0], "v40a": [0, .25, .5, 1, 1.5, 2, 3], "free": [0, .25, .5, 1, 1.5, 2, 3]}


def run(gname, src, mode):
    groups = [[I[l] for l in g] for g in GROUPS[gname]]
    Ss = [SPEC[(gname, src, tuple(g))] for g in groups]

    def build(alist):
        lp = base
        for g, S, a in zip(groups, Ss, alist):
            lp = apply(lp, g, S, mode, a)
        return lp

    # 묶음별 alpha 좌표 하강(전체 OOF로) = 낙관치
    def fit(mask):
        al = [0.0] * len(groups)
        for _ in range(2):
            for gi in range(len(groups)):
                best = None
                for a in ALPHAS[mode]:
                    al2 = al.copy(); al2[gi] = a
                    lp = build(al2)
                    p = lp.argmax(1)
                    s = np.mean([(p[mask & (y == k)] == k).mean() for k in range(K)])
                    if best is None or s > best[0] + 1e-12:
                        best = (s, a)
                al[gi] = best[1]
        return al

    al_all = fit(np.ones(N, bool))
    opt = ba(build(al_all))
    # nested: fold k의 alpha는 나머지 fold로 고름
    final = base.copy()
    for k in range(5):
        al = fit(fold != k)
        m = fold == k
        final[m] = build(al)[m]
    return opt, ba(final), al_all, final


SPEC = {}
for gname, gl in GROUPS.items():
    for g in gl:
        cols = [I[l] for l in g]
        for src in ["stack", "text"]:
            SPEC[(gname, src, tuple(cols))] = specialist(cols, src)
        # 특화 분류기 단독의 묶음 내부 정확도 vs v4 내부 정확도
        m = np.isin(y, cols)
        yc = np.array([cols.index(v) for v in y[m]])
        v4in = (base[m][:, cols].argmax(1) == yc).mean()
        print(f"[{gname}] {'/'.join(g)}: 내부정확도 v4 {v4in:.3f} | stack {(SPEC[(gname,'stack',tuple(cols))][m].argmax(1)==yc).mean():.3f} | text {(SPEC[(gname,'text',tuple(cols))][m].argmax(1)==yc).mean():.3f}")

print(f"\nv4 OOF BA {ba(base):.4f}")
rows = []
for gname in GROUPS:
    for src in ["stack", "text"]:
        for mode in ["gate", "v40a", "free"]:
            opt, nest, al, final = run(gname, src, mode)
            rows.append((gname, src, mode, opt, nest, al))
            print(f"{gname:9s} {src:5s} {mode:4s}  best-alpha {opt:.4f} ({opt-ba(base):+.4f})  nested {nest:.4f} ({nest-ba(base):+.4f})  alpha={al}")
