"""회사 게이트: 특화 분류기 입력(stack: n-gram+LoRA / stack+svc: + 회사 전용 SVC) 비교, nested alpha, 프롬프트 bootstrap."""
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, log_softmax
from sklearn.linear_model import LogisticRegression

ROOT, FS = Path(sys.argv[1]), np.load(sys.argv[2])
d = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
L = [str(x) for x in d["labels"]]
I = {l: i for i, l in enumerate(L)}
y, fold, base, ng, lo, pid = d["y"], d["fold"], d["eval_lp"], d["ng_oof_lp"], d["lora_lp"], d["prompt_id"]
N, K = base.shape
FAMS = {p: [I[l] for l in L if l.startswith(p)] for p in ["gpt", "gemini", "claude", "glm"]}


def ba(lp, mask=None):
    p = lp.argmax(1)
    m = np.ones(N, bool) if mask is None else mask
    return np.mean([(p[m & (y == k)] == k).mean() for k in range(K)])


def spec(name, cols, feats, C):
    X = np.hstack(feats)
    out = np.zeros((N, len(cols)))
    inC = np.isin(y, cols)
    yc = np.array([cols.index(v) if v in cols else -1 for v in y])
    for k in range(5):
        tr = (fold != k) & inC
        m = LogisticRegression(C=C, max_iter=3000).fit(X[tr], yc[tr])
        out[fold == k] = m.predict_log_proba(X[fold == k])
    return out


def gate(lp, cols, S, a):
    lp = lp.copy()
    sub = lp[:, cols]
    mass = logsumexp(sub, 1, keepdims=True)
    new = log_softmax((1 - a) * (sub - mass) + a * S, axis=1) + mass
    g = np.isin(lp.argmax(1), cols)[:, None]
    lp[:, cols] = np.where(g, new, sub)
    return lp


ALPHAS = [0, .25, .5, .75, 1.0]


def evaluate(Ss, label):
    cols_l = list(FAMS.values())

    def build(al):
        lp = base
        for c, S, a in zip(cols_l, Ss, al):
            lp = gate(lp, c, S, a)
        return lp

    def fit(mask):
        al = [0.0] * 4
        for _ in range(2):
            for i in range(4):
                sc = []
                for a in ALPHAS:
                    t = al.copy(); t[i] = a
                    sc.append(ba(build(t), mask))
                al[i] = ALPHAS[int(np.argmax(sc))]
        return al

    al = fit(None)
    final = base.copy()
    nal = []
    for k in range(5):
        a = fit(fold != k); nal.append(a)
        final[fold == k] = build(a)[fold == k]
    print(f"{label:22s} best-alpha {ba(build(al)):.4f} ({ba(build(al))-ba(base):+.4f}) alpha={al} | nested {ba(final):.4f} ({ba(final)-ba(base):+.4f})")
    return final, build([1.0] * 4)


res = {}
for C in [0.1, 1.0]:
    for src in ["stack", "stack+svc"]:
        Ss = []
        for name, cols in FAMS.items():
            feats = [ng[:, cols], lo[:, cols]]
            if src == "stack+svc":
                sv = FS[name]
                feats.append(sv)
            Ss.append(spec(name, cols, feats, C))
        inner = []
        for (name, cols), S in zip(FAMS.items(), Ss):
            m = np.isin(y, cols)
            yc = np.array([cols.index(v) for v in y[m]])
            inner.append(f"{name} {(S[m].argmax(1)==yc).mean():.3f}")
        print(f"[C={C} {src}] 내부정확도 " + ", ".join(inner))
        res[(C, src)] = evaluate(Ss, f"C={C} {src}")

# bootstrap: 프롬프트 단위, nested 결과 기준
rng = np.random.default_rng(0)
P = np.unique(pid)
idx_by_p = {p: np.where(pid == p)[0] for p in P}
for key, (final, _) in res.items():
    diffs = []
    for _ in range(1000):
        ps = rng.choice(P, len(P), replace=True)
        ii = np.concatenate([idx_by_p[p] for p in ps])
        yy = y[ii]
        f = lambda lp: np.mean([(lp[ii].argmax(1)[yy == k] == k).mean() for k in range(K)])
        diffs.append(f(final) - f(base))
    lo_, hi_ = np.percentile(diffs, [2.5, 97.5])
    # 바뀐 지문 수, 회사 수준 정확도 변화
    ch = (final.argmax(1) != base.argmax(1)).sum()
    fix = ((final.argmax(1) == y) & (base.argmax(1) != y)).sum()
    brk = ((final.argmax(1) != y) & (base.argmax(1) == y)).sum()
    print(f"{key}: nested dBA 95% [{lo_:+.4f}, {hi_:+.4f}] | 바뀐 지문 {ch} (고침 {fix}, 망침 {brk})")
