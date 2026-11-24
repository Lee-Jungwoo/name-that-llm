"""쌍별 구분 가능성: 두 모델만 가르는 이진 분류기 AUC와 v4 18-class 모델의 같은 쌍 AUC를 비교한다.

- 전용 분류기: Open Pool의 A, B 지문(각 2,725개)으로 학습하고 train 지문(각 약 145개)으로 평가한다.
  풀과 train은 프롬프트가 겹치지 않으므로 주제 누수가 없다. train과 같은 지문은 풀에서 뺀다.
  (1) 문체 피처 27개 + 로지스틱 회귀, (2) 문자 1-4gram + 단어 1-2gram TF-IDF + 로지스틱 회귀.
- v4: 블렌드 log 확률의 차이 lp[A] - lp[B]로 A, B 지문을 줄 세운 AUC. 5-fold OOF 전체(각 약 150개).
- 짝 정확도: 같은 프롬프트에 대한 A, B 답 두 개를 주고 어느 쪽이 A인지 맞히는 비율(점수 차이 비교).
출력: out/separability.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, LABELS, feature_frame, load_pool, load_train, pool_features  # noqa: E402
from paired import OUT, pairs  # noqa: E402

norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()


def paired_acc(s, df_ab, a):
    """같은 프롬프트의 A, B 답 중 점수가 높은 쪽을 A로 고를 때의 정답률."""
    t = df_ab.assign(s=s).pivot_table(index="prompt_id", columns="model", values="s")
    t = t.dropna()
    return ((t[a] > t.drop(columns=a).iloc[:, 0]).mean(), len(t))


def main():
    df, d = load_train()
    lp = d["eval_lp"]
    has_text = df.text.notna().values
    tr = df[has_text].reset_index(drop=True)
    Ftr = feature_frame(tr.text)

    pool = load_pool()
    P = pool_features()
    keep = ~pool.text.map(norm).isin(set(tr.text.map(norm)))
    print("train과 겹쳐서 뺀 풀 지문:", (~keep).sum())
    pool, P = pool[keep].reset_index(drop=True), P[keep].reset_index(drop=True)

    rows = []
    for org, a, b in pairs():
        ia, ib = LABELS.index(a), LABELS.index(b)
        # v4 18-class 모델(OOF 전체)
        m = np.isin(df.y.values, [ia, ib])
        s_v4 = lp[m, ia] - lp[m, ib]
        auc_v4 = roc_auc_score(df.y.values[m] == ia, s_v4)
        acc_v4, n_v4 = paired_acc(s_v4, df.loc[m, ["prompt_id", "model"]], a)
        # v4가 둘 중 하나로 예측한 지문 중 상대로 잘못 간 비율(18-class argmax 기준)
        pm = df.pred.values[m]
        conf = ((df.y.values[m] == ia) & (pm == ib)).sum() + ((df.y.values[m] == ib) & (pm == ia)).sum()

        pm_ = pool.model.isin([a, b]).values
        ytr_pool = (pool.model.values[pm_] == a).astype(int)
        tm = tr.model.isin([a, b]).values
        yte = (tr.model.values[tm] == a).astype(int)
        out = dict(org=org, a=a, b=b, n_test=tm.sum(), auc_v4=auc_v4, pair_acc_v4=acc_v4, n_prompt=n_v4,
                   v4_cross_errors=conf, v4_n=m.sum())

        # (1) 문체 피처
        Xp = np.log1p(P.loc[pm_, list(FEATURES)].clip(lower=0).values)
        Xt = np.log1p(Ftr.loc[tm, list(FEATURES)].clip(lower=0).values)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xp, ytr_pool)
        s = clf.decision_function(Xt)
        out["auc_stylo"] = roc_auc_score(yte, s)
        out["pair_acc_stylo"] = paired_acc(s, tr.loc[tm, ["prompt_id", "model"]], a)[0]

        # (2) n-gram
        cv = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3, sublinear_tf=True, lowercase=False,
                             max_features=200000)
        wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                             token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=100000)
        tp = pool.text.values[pm_]
        Xp = sp.hstack([cv.fit_transform(tp), wv.fit_transform(tp)]).tocsr()
        tt = tr.text.values[tm]
        Xt = sp.hstack([cv.transform(tt), wv.transform(tt)]).tocsr()
        clf = LogisticRegression(C=4.0, max_iter=3000).fit(Xp, ytr_pool)
        s = clf.decision_function(Xt)
        out["auc_ngram"] = roc_auc_score(yte, s)
        out["pair_acc_ngram"] = paired_acc(s, tr.loc[tm, ["prompt_id", "model"]], a)[0]
        rows.append(out)
        print(f"[{org}] {a} vs {b}: v4 {auc_v4:.3f} / 문체 {out['auc_stylo']:.3f} / n-gram {out['auc_ngram']:.3f}"
              f" | 짝 정확도 v4 {acc_v4:.3f}, n-gram {out['pair_acc_ngram']:.3f} | v4 상호 오답 {conf}/{m.sum()}")
    pd.DataFrame(rows).to_csv(OUT / "separability.csv", index=False)


if __name__ == "__main__":
    main()
