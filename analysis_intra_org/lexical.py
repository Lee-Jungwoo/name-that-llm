"""헷갈리는 쌍의 어휘 차이: 정보 사전분포를 둔 log-odds ratio(Monroe et al., 2008)의 z 점수.

- Open Pool(쌍마다 지문 5,450개)에서 단어별 z를 구하고, train(쌍마다 지문 약 290개)에서 같은 방향인지 본다.
- 사전분포는 18개 모델 풀 전체의 단어 빈도.
- 지문 빈도(그 단어가 나온 지문 비율)를 함께 적는다. 한 지문에서 반복된 단어에 끌려가지 않게 하려는 것.
출력: out/lexical.csv
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_pool, load_train  # noqa: E402
from paired import OUT  # noqa: E402

TOK = re.compile(r"[a-z’']+|[—;:,()\"“”]")
HARD = [("gpt-5.6-sol", "gpt-5.6-terra"), ("gpt-5.6-luna", "gpt-5.6-sol"), ("gpt-5.6-luna", "gpt-5.6-terra"),
        ("gemini-3.7-flash", "gemini-3.8-flash"), ("gemini-3.6-flash", "gemini-3.7-flash"),
        ("glm-5.3", "glm-5.3-flash"), ("claude-fable-5", "glm-5.3-flash"), ("claude-fable-5", "claude-opus-4-8"),
        ("claude-opus-4-8", "claude-sonnet-5"), ("kimi-k3", "gpt-5.5")]


def doc_sets(texts):
    return [set(TOK.findall(t.lower())) for t in texts]


def counts(texts):
    c = Counter()
    for t in texts:
        c.update(TOK.findall(t.lower()))
    return c


def log_odds(ca, cb, prior, a0=500.0):
    """Monroe et al. 식 (16)~(22). prior는 전체 빈도, a0는 사전분포 세기."""
    vocab = [w for w in prior if ca[w] + cb[w] >= 20]
    tot = sum(prior.values())
    na, nb = sum(ca.values()), sum(cb.values())
    rows = []
    for w in vocab:
        aw = a0 * prior[w] / tot
        la = np.log((ca[w] + aw) / (na + a0 - ca[w] - aw))
        lb = np.log((cb[w] + aw) / (nb + a0 - cb[w] - aw))
        var = 1 / (ca[w] + aw) + 1 / (cb[w] + aw)
        rows.append((w, (la - lb) / np.sqrt(var)))
    return pd.DataFrame(rows, columns=["token", "z"])


def main():
    pool = load_pool()
    tr, _ = load_train()
    tr = tr[tr.text.notna()]
    prior = counts(pool.text)
    out = []
    for a, b in HARD:
        pa, pb = pool.text[pool.model == a], pool.text[pool.model == b]
        L = log_odds(counts(pa), counts(pb), prior)
        da, db = doc_sets(pa), doc_sets(pb)
        ta, tb = doc_sets(tr.text[tr.model == a]), doc_sets(tr.text[tr.model == b])
        L["pool_df_a"] = [np.mean([w in s for s in da]) for w in L.token]
        L["pool_df_b"] = [np.mean([w in s for s in db]) for w in L.token]
        L["train_df_a"] = [np.mean([w in s for s in ta]) for w in L.token]
        L["train_df_b"] = [np.mean([w in s for s in tb]) for w in L.token]
        L["replicated"] = np.sign(L.z) == np.sign(L.train_df_a - L.train_df_b)
        L["a"], L["b"] = a, b
        out.append(L)
        top = L.reindex(L.z.abs().sort_values(ascending=False).index).head(40)
        rep = top.replicated.mean()
        print(f"\n{a} vs {b}: |z| 상위 40개 중 train 재현 {rep:.0%}, |z|>3 단어 {int((L.z.abs() > 3).sum())}개")
        for side, g in [(a, L[L.z > 0].nlargest(8, "z")), (b, L[L.z < 0].nsmallest(8, "z"))]:
            s = ", ".join(f"{r.token}({r.z:+.1f}, 풀 {r.pool_df_a:.0%}/{r.pool_df_b:.0%}, train {r.train_df_a:.0%}/{r.train_df_b:.0%})"
                          for r in g.itertuples())
            print(f"  {side} 쪽: {s}")
    pd.concat(out).to_csv(OUT / "lexical.csv", index=False)


if __name__ == "__main__":
    main()
