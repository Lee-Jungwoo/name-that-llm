"""회사별 n-gram 특화 분류기(풀 + train) 5-fold OOF를 만든다. v4 n-gram과 같은 4뷰 특성.
사용법: python famsvc.py <repo> <hf parquet 폴더> <출력 npz>"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT, HF, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
T0 = time.time()
d = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
L = [str(x) for x in d["labels"]]
I = {l: i for i, l in enumerate(L)}
y, fold = d["y"], d["fold"]
t = pd.read_csv(ROOT / "data" / "train.csv")
key = pd.DataFrame({"prompt_id": d["prompt_id"], "model": [L[i] for i in y]})
text = key.merge(t, on=["prompt_id", "model"], how="left").text.astype(str).values

g = pd.concat([pd.read_parquet(HF / f"{s}-00000-of-00001.parquet")
               for s in ["train_stage1", "train_stage2", "val_holdout", "clean_pool"]])
pool = g[(g.source == "open_pool") & g.model.isin(L)].drop_duplicates("id")
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
pool = pool[~pool.text.map(norm).isin(set(map(norm, text)))]
pool = pool.sample(frac=1, random_state=42).groupby("model").head(2500)
ptext, py = pool.text.astype(str).values, pool.model.map(I).values
print("pool", len(pool), np.bincount(py, minlength=18).min())

FW = set("""a an the and or but if while although because as so than that which who whom whose this these
those it its they them their he him his she her we us our you your i me my of in on at by for with from to into over
under about between through during not no nor is are was were be been being has have had do does did will would can
could may might must shall should very more most such only own same just also""".split())
shape = lambda s: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", s)
func = lambda s: " ".join(w.lower() if w.lower() in FW or not w.isalnum() else "_" for w in re.findall(r"\w+|[^\w\s]", s))
VIEWS = [
    (None, dict(analyzer="char", ngram_range=(3, 5), max_features=250_000), 1.0),
    (None, dict(analyzer="word", token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), max_features=150_000), 0.7),
    (shape, dict(analyzer="char", ngram_range=(2, 6), max_features=60_000), 0.5),
    (func, dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3), max_features=50_000), 0.5),
]
Xp, Xt = [], []
for fn, opts, w in VIEWS:
    v = TfidfVectorizer(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32, **opts)
    prep = (lambda a: [fn(s) for s in a]) if fn else list
    Xp.append(v.fit_transform(prep(ptext)) * w)
    Xt.append(v.transform(prep(text)) * w)
Xp, Xt = sparse.hstack(Xp).tocsr(), sparse.hstack(Xt).tocsr()
print("features", Xp.shape, f"{time.time()-T0:.0f}s")

FAMS = {p: [I[l] for l in L if l.startswith(p)] for p in ["gpt", "gemini", "claude", "glm"]}
res = {}
for name, cols in FAMS.items():
    out = np.zeros((len(y), len(cols)))
    pm = np.isin(py, cols)
    for k in range(5):
        tr = np.where((fold != k) & np.isin(y, cols))[0]
        X = sparse.vstack([Xp[pm], Xt[tr]]).tocsr()
        yy = np.concatenate([py[pm], y[tr]])
        ww = np.concatenate([np.ones(pm.sum()), np.full(len(tr), 12.0)])
        m = LinearSVC(C=0.1).fit(X, yy, sample_weight=ww)
        te = fold == k
        sc = m.decision_function(Xt[te])
        if sc.ndim == 1:  # 2-클래스
            sc = np.stack([-sc, sc], 1)
        out[te] = sc[:, [list(m.classes_).index(c) for c in cols]]
    res[name] = out
    inC = np.isin(y, cols)
    yc = np.array([cols.index(v) for v in y[inC]])
    v4 = (d["eval_lp"][inC][:, cols].argmax(1) == yc).mean()
    ng = (d["ng_oof_lp"][inC][:, cols].argmax(1) == yc).mean()
    print(f"{name}: 내부정확도 v4블렌드 {v4:.3f} | v4 n-gram {ng:.3f} | 회사전용 SVC {(out[inC].argmax(1)==yc).mean():.3f} | {time.time()-T0:.0f}s")
np.savez(OUT, **res)
