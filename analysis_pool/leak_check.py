"""Open Pool에 train 프롬프트(또는 그와 아주 가까운 주제)에 답한 지문이 있는지 확인합니다.

테스트 프롬프트는 train에 없으므로, 검증 fold의 프롬프트와 같은 프롬프트로 쓴 풀 지문이 학습 행에 섞여 있으면
OOF가 리더보드보다 좋게 나옵니다. 표본을 키울수록 그런 행이 늘면 표본 크기 효과 자체가 그 때문일 수 있습니다.

1. 풀이 스스로 밝힌 값: prompt_canon(풀 안의 프롬프트 id), flag_near_train_prompt, flag_train_prompt_regen,
   max_cos_train(train 지문과의 최대 TF-IDF 코사인), max_sim_train_prompt(train 프롬프트 중심과의 최대 코사인).
2. 독립 확인: 내용어 TF-IDF로 train 프롬프트 150개의 중심을 만들고, 풀 지문마다 가장 가까운 train 프롬프트와 그 코사인을
   구합니다. 기준은 train 안에서 잽니다: 지문 하나를 빼고 만든 자기 프롬프트 중심과의 코사인(같은 프롬프트)과,
   다른 프롬프트 중심 중 가장 가까운 것과의 코사인(다른 프롬프트).
3. 표본 크기별(sample_pool(n))로 같은 수를 세고, fit에서 행을 뺄 때 쓰도록 out/pool/cache_n<n>/leak.npz에 저장합니다
   (표본 행마다 train 프롬프트 150개와의 코사인 전부).

사용법: .venv/bin/python -I -W ignore analysis_pool/leak_check.py [n ...] [--samples] > analysis_pool/logs/leak_check.log
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis_corpus"))
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common import LABELS, L2I, ROOT, load_pool, load_train, sample_pool, stylo_raw

OUT = ROOT / "out" / "pool"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
sizes = [int(a) for a in sys.argv[1:] if a.isdigit()] or [2500, 3500, 5000, 7500, 10000]

train = load_train()
pool = load_pool(["text", "model", "match_weight", "prompt_canon", "prompt_group", "prompt_evidence", "source_dataset",
                  "max_cos_train", "max_sim_train_prompt", "flag_near_train_prompt", "flag_train_prompt_regen"])
print(f"train {len(train):,}개 지문, 프롬프트 {train.prompt_id.nunique()}개 | Open Pool {len(pool):,}개")

print("\n== 1. 풀이 밝힌 값 ==")
canon = pool.prompt_canon.fillna("")
print("train prompt_id 예:", sorted(train.prompt_id.unique())[:3], "| 풀 prompt_canon 예:", canon[canon != ""].unique()[:4].tolist())
print("prompt_canon 첫 글자 분포:", canon.str[:1].replace("", "(없음)").value_counts().to_dict())
print("train prompt_id와 글자 그대로 같은 prompt_canon:", int(canon.isin(set(train.prompt_id)).sum()), "행 |",
      "prompt_group 끝이 train prompt_id와 같은 행:", int(pool.prompt_group.fillna("").str.split(":").str[-1].isin(set(train.prompt_id)).sum()))
print("flag_near_train_prompt:", int(pool.flag_near_train_prompt.sum()), "| flag_train_prompt_regen:", int(pool.flag_train_prompt_regen.sum()))
print(pool[["max_cos_train", "max_sim_train_prompt"]].describe(percentiles=[.5, .9, .99, .999]).T.round(3).to_string())

print("\n== 2. 독립 확인: 내용어 TF-IDF로 본 train 프롬프트와의 거리 ==")
vec = TfidfVectorizer(lowercase=True, stop_words="english", token_pattern=r"[a-z]{3,}", sublinear_tf=True, min_df=2, dtype=np.float32)
vec.fit(pd.concat([train.text, pool.text]))
Xt, Xp = vec.transform(train.text), vec.transform(pool.text)
pids = np.array(sorted(train.prompt_id.unique()))
pidx = np.searchsorted(pids, train.prompt_id.values)
sums = np.zeros((len(pids), Xt.shape[1]), dtype=np.float32)
np.add.at(sums, pidx, Xt.toarray())
unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
C = unit(sums)
# train 기준: 자기 지문을 뺀 자기 프롬프트 중심과의 코사인, 다른 프롬프트 중심 중 최대
Xtd = Xt.toarray()
own = np.einsum("ij,ij->i", Xtd, unit(sums[pidx] - Xtd))
other = Xtd @ C.T
other[np.arange(len(train)), pidx] = -1
other_max = other.max(1)
q = lambda a, ps: {p: round(float(np.percentile(a, p)), 3) for p in ps}
print("train 지문 대 자기 프롬프트(자기 제외):", q(own, [1, 5, 25, 50, 75]))
print("train 지문 대 가장 가까운 다른 프롬프트:", q(other_max, [50, 95, 99, 99.9, 100]))
TAU1, TAU5 = float(np.percentile(own, 1)), float(np.percentile(own, 5))
S = np.asarray(Xp @ C.T, dtype=np.float32)
sim, nn = S.max(1), S.argmax(1)
del Xtd
print("풀 지문 대 가장 가까운 train 프롬프트:", q(sim, [50, 95, 99, 99.9, 100]))
print(f"기준: 같은 프롬프트 train 지문의 1% 분위 {TAU1:.3f}, 5% 분위 {TAU5:.3f}")
for name, tau in (("1% 분위", TAU1), ("5% 분위", TAU5), ("다른 프롬프트 99% 분위", float(np.percentile(other_max, 99)))):
    print(f"  코사인 >= {tau:.3f} ({name}): 풀 {int((sim >= tau).sum()):,}행 ({(sim >= tau).mean():.4f}) | "
          f"train에서 다른 프롬프트인데 넘는 지문 {int((other_max >= tau).sum())}개 ({(other_max >= tau).mean():.4f})")
pool["sim"], pool["nn_prompt"], pool["row"] = sim, pids[nn], np.arange(len(pool))
print("풀이 밝힌 max_sim_train_prompt와 이 코사인의 Spearman:", round(float(pool[["sim", "max_sim_train_prompt"]].corr(method="spearman").iloc[0, 1]), 3))
# 출처가 붙인 프롬프트 id가 train의 prompt_id와 같은 형식인 행: 같은 번호의 train 프롬프트와 실제로 가까운지 봅니다.
tail = pool.prompt_group.fillna("").str.split(":").str[-1]
pool["named"] = tail.isin(set(pids)).values
named = pool[pool.named]
named_sim = S[named.row.values, np.searchsorted(pids, tail[named.index].values)]
print(f"\n출처의 프롬프트 id가 train prompt_id와 같은 풀 지문 {len(named)}개 (출처 {named.source_dataset.value_counts().to_dict()}, "
      f"train 프롬프트 {tail[named.index].nunique()}개): 같은 번호의 train 프롬프트 중심과의 코사인 {q(named_sim, [0, 25, 50, 75, 100])} "
      f"| 풀이 밝힌 max_sim_train_prompt {q(named.max_sim_train_prompt.values, [0, 50, 100])}")
print(f"  이 가운데 가장 가까운 train 프롬프트가 바로 그 번호인 행: {int((named.nn_prompt.values == tail[named.index].values).sum())}개")
hit = pool[pool.sim >= TAU1]
print(f"\n코사인 >= {TAU1:.3f}인 풀 지문 {len(hit):,}개: 라벨별", hit.groupby("model").size().reindex(LABELS, fill_value=0).to_dict())
print("  걸린 train 프롬프트 수:", hit.nn_prompt.nunique(), "| 프롬프트당 행 수 상위:", hit.nn_prompt.value_counts().head(8).to_dict())
print("  출처 상위:", hit.source_dataset.value_counts().head(6).to_dict())
print("  prompt_evidence:", hit.prompt_evidence.value_counts().to_dict(), "| 서로 다른 prompt_canon", hit.prompt_canon.nunique())
print("  가장 가까운 5개 (코사인, 라벨, 가까운 train 프롬프트, 출처):")
for _, r in pool.nlargest(5, "sim").iterrows():
    print(f"    {r.sim:.3f} {r.model} {r.nn_prompt} {r.source_dataset}")
    if "--samples" in sys.argv:  # 대회 지문을 저장소 로그에 남기지 않으려고 기본으로는 출력하지 않음
        print(f"      풀: {' '.join(r.text.split()[:12])} / train: {' '.join(train.text[train.prompt_id == r.nn_prompt].iloc[0].split()[:12])}")

print("\n== 3. 표본 크기별 ==")
rows = []
for n in sizes:
    s = sample_pool(pool, n)
    cache = OUT / f"cache_n{n}"
    aligned = None
    if (cache / "y_pool.npy").exists():  # fit에 쓴 표본과 같은 행·같은 순서인지 확인하고 저장
        yp, Sp = np.load(cache / "y_pool.npy"), np.load(cache / "S_pool.npy")
        pick = np.random.default_rng(0).choice(len(s), 40, replace=False)
        aligned = bool(len(s) == len(yp) and (s.model.map(L2I).values == yp).all()
                       and np.array_equal(stylo_raw(s.text.iloc[pick]), Sp[pick]))
        assert aligned, f"n={n}: 다시 뽑은 표본이 저장된 표본과 다릅니다"
        np.savez_compressed(cache / "leak.npz", sim=s.sim.values.astype(np.float32), nn_prompt=s.nn_prompt.values.astype(str),
                            bsim=s.max_sim_train_prompt.values.astype(np.float32), bcos=s.max_cos_train.values.astype(np.float32),
                            tau1=TAU1, tau5=TAU5, prompt_ids=pids, sim_all=S[s.row.values].astype(np.float16))
    per_prompt = s.nn_prompt.value_counts().reindex(pids, fill_value=0)
    hit = s[s.sim >= TAU1]
    rows.append({"라벨당": n, "행": len(s), "저장된 표본과 일치": aligned,
                 f"코사인>={TAU1:.2f} 행": len(hit), "비율": len(hit) / len(s),
                 f"코사인>={TAU5:.2f} 행": int((s.sim >= TAU5).sum()), "그 비율": float((s.sim >= TAU5).mean()),
                 "train 프롬프트당 (τ1) 평균": len(hit) / len(pids), "걸린 프롬프트 수": hit.nn_prompt.nunique(),
                 "출처 id가 train prompt_id와 같은 행": int(s.named.sum()),
                 "코사인 평균": s.sim.mean(), "코사인 99%": float(np.percentile(s.sim, 99)),
                 "max_sim_train_prompt 평균": s.max_sim_train_prompt.mean(), "max_cos_train 평균": s.max_cos_train.mean(),
                 "가장 가까운 프롬프트 기준 프롬프트당 행": per_prompt.mean(), "그 최대": int(per_prompt.max())})
    v4 = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
    Ss = S[s.row.values]
    for tau in (TAU1, 0.10):
        share = [float((Ss[:, np.unique(pidx[v4["fold"] == k])].max(1) >= tau).mean()) for k in range(5)]
        print(f"n={n}: fold의 검증 프롬프트 중 하나와 코사인 >= {tau:.3f}인 풀 행의 비율(fold별) {np.round(share, 3).tolist()}")
    print(f"n={n}: 코사인 >= {TAU1:.3f}인 행의 라벨별 수", hit.groupby("model").size().reindex(LABELS, fill_value=0).to_dict())
print(pd.DataFrame(rows).set_index("라벨당").T.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\n전체 풀: 코사인 >= {TAU1:.3f} 비율 {(pool.sim >= TAU1).mean():.4f}, >= {TAU5:.3f} 비율 {(pool.sim >= TAU5).mean():.4f}")

print("\n== 4. 지문 단위: 풀 안에 '주제가 가까운 이웃'이 얼마나 있는가 (train 지문 대 공개된 test 지문 3개) ==")
# 풀이 train 프롬프트 쪽으로만 쏠려 있다면, train 지문은 풀에서 가까운 이웃을 많이 찾고 test 지문은 못 찾을 것입니다.
# test.csv에 공개된 지문은 3개뿐이라 참고용입니다.
test = pd.read_csv(ROOT / "data" / "name-that-llm" / "test.csv")
Xs = vec.transform(test.text.fillna("").astype(str))


def neighbours(X):
    mx, c2, c3 = [], [], []
    for i in range(0, X.shape[0], 200):
        D = (X[i:i + 200] @ Xp.T).toarray()
        mx.append(D.max(1)); c2.append((D >= 0.2).sum(1)); c3.append((D >= 0.3).sum(1))
    return np.concatenate(mx), np.concatenate(c2), np.concatenate(c3)


tm, t2, t3 = neighbours(Xt)
sm, s2, s3 = neighbours(Xs)
print("train 지문 2,699개: 풀 지문과의 최대 코사인", q(tm, [5, 25, 50, 75, 95]), "| 코사인 >= 0.2인 풀 지문 수", q(t2, [5, 25, 50, 75, 95]),
      "| >= 0.3", q(t3, [5, 25, 50, 75, 95]))
for i in range(len(test)):
    print(f"test 지문 {i}: 최대 코사인 {sm[i]:.3f} (train 분포에서 {float((tm < sm[i]).mean()) * 100:.0f}% 분위), "
          f"코사인 >= 0.2인 풀 지문 {int(s2[i])}개 ({float((t2 < s2[i]).mean()) * 100:.0f}% 분위), >= 0.3 {int(s3[i])}개")
# train 지문끼리(다른 프롬프트)와 비교: 같은 주최자가 만든 다른 프롬프트의 지문과는 얼마나 가까운가
D = (Xt @ Xt.T).toarray()
D[pidx[:, None] == pidx[None, :]] = -1
print("train 지문 대 다른 프롬프트의 train 지문: 최대 코사인", q(D.max(1), [5, 25, 50, 75, 95]))
