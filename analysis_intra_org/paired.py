"""같은 프롬프트에 대한 두 모델의 답을 짝지어 문체 피처 차이를 잰다.

- train은 프롬프트마다 18개 모델이 한 번씩 답했으므로, 두 모델의 차이를 주제 영향 없이 비교할 수 있다.
- 효과 크기 dz = 짝 차이의 평균 / 짝 차이의 표준편차. |dz| 0.2 작음, 0.5 중간, 0.8 큼.
- 같은 방향 비율 = 두 값이 다른 짝 중 A > B인 비율.
- Open Pool(모델당 2,725개, 짝 없음)에서 같은 방향인지(Cohen's d 부호)로 재현을 확인한다.
출력: out/paired.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, LABELS, ORGS, feature_frame, load_train, pool_features  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CROSS = [("claude-fable-5", "glm-5.3-flash"), ("claude-fable-5", "glm-5.3"), ("kimi-k3", "gpt-5.5")]


def pairs():
    for org, ms in ORGS.items():
        for a, b in itertools.combinations(ms, 2):
            yield org, a, b
    for a, b in CROSS:
        yield "cross", a, b


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    df, _ = load_train()
    df = df[df.text.notna()].reset_index(drop=True)
    F = pd.concat([df[["prompt_id", "model"]], feature_frame(df.text)], axis=1)
    P = pool_features()

    rows = []
    for org, a, b in pairs():
        A = F[F.model == a].set_index("prompt_id")
        B = F[F.model == b].set_index("prompt_id")
        common = A.index.intersection(B.index)
        PA, PB = P[P.model == a], P[P.model == b]
        for f in FEATURES:
            d = A.loc[common, f].values - B.loc[common, f].values
            nz = d[d != 0]
            sd = d.std(ddof=1)
            dz = d.mean() / sd if sd > 0 else 0.0
            p = wilcoxon(nz).pvalue if len(nz) >= 10 else 1.0
            sp = np.sqrt((PA[f].var() + PB[f].var()) / 2)
            dpool = (PA[f].mean() - PB[f].mean()) / sp if sp > 0 else 0.0
            rows.append(dict(org=org, a=a, b=b, feature=f, n_pairs=len(common),
                             mean_a=A.loc[common, f].mean(), mean_b=B.loc[common, f].mean(),
                             dz=dz, a_gt_b=(nz > 0).mean() if len(nz) else np.nan, n_nonzero=len(nz),
                             p=p, pool_d=dpool))
    R = pd.DataFrame(rows)
    R["q"] = bh(R.p.values)
    R["replicated"] = np.sign(R.dz) == np.sign(R.pool_d)
    R.to_csv(OUT / "paired.csv", index=False)

    pd.set_option("display.width", 220)
    for (org, a, b), g in R.groupby(["org", "a", "b"], sort=False):
        n = g.n_pairs.iloc[0]
        g = g[(g.q < 0.05) & g.replicated]
        g = g.loc[g.dz.abs().sort_values(ascending=False).index]
        print(f"\n[{org}] {a} vs {b}  (짝 {n}개, 유의+재현 피처 {len(g)}개)")
        if len(g):
            print(g.head(6)[["feature", "mean_a", "mean_b", "dz", "a_gt_b", "q", "pool_d"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
