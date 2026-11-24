"""Response Corpus의 short 뷰를 살펴봅니다: 라벨 대응, 라벨별 행 수, 길이, 출처, 중복, 장르, 문체 피처.

긴 OOF 실행 전에 "train과 얼마나 닮았는가"를 싸게 확인하는 단계입니다. 결과는 표준 출력으로만 냅니다.

사용법: .venv/bin/python -I -W ignore analysis_corpus/inspect_corpus.py [--samples] > analysis_corpus/logs/inspect.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import (LABELS, L2I, MARKDOWN_RE, REFUSAL_RE, ROOT, STYLO_NAMES, load_corpus, load_pool, load_train, norm,
                    sample_pool, stylo_raw)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")


def head(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}", flush=True)


corpus = load_corpus()
train = load_train()
pool = load_pool()
train["norm"] = train.text.map(norm)
pool_norm = set(pool.text.map(norm))

head("1. 라벨 대응")
cols = sorted(corpus.model.unique())
print("corpus 모델 열:", cols)
print("프로젝트 LABELS에 없는 열:", sorted(set(cols) - set(LABELS)), "| corpus에 없는 라벨:", sorted(set(LABELS) - set(cols)))

head("2. 중복")
print(f"셀 {len(corpus):,}개, response_id {corpus.response_id.nunique():,}개, 프롬프트 그룹 {corpus.prompt_group_id.nunique():,}개")
dup_in = corpus.duplicated(["model", "norm"])
cross = corpus.groupby("norm").model.nunique()
print(f"같은 라벨 안 정규화 중복 {int(dup_in.sum())}개, 서로 다른 라벨에 같은 정규화 텍스트 {int((cross > 1).sum())}종")
in_train = corpus.norm.isin(set(train.norm))
in_pool = corpus.norm.isin(pool_norm)
print(f"train과 정규화 텍스트가 같은 셀 {int(in_train.sum())}개, Open Pool과 같은 셀 {int(in_pool.sum())}개")
if in_pool.any():
    print(corpus[in_pool].groupby(["source", "model"]).size().to_string())
# 긴 응답 안에 train 지문이 통째로 들어 있는지(부분 문자열)
tn = [t for t in train.norm if len(t) > 80]
hits = sum(any(t in c for t in tn) for c in corpus.norm[corpus.norm.str.len() > 400].sample(2000, random_state=0))
print(f"긴 셀 2,000개 표본 중 train 지문을 통째로 포함한 셀 {hits}개")

head("3. 라벨별 행 수와 길이 (공백 기준 단어 수)")
tab = corpus.groupby("model").ws_words.describe()[["count", "min", "25%", "50%", "75%", "max"]].reindex(LABELS)
tab["train_n"] = train.groupby("model").size().reindex(LABELS)
tab["train_50%"] = train.groupby("model").text.apply(lambda s: s.str.split().str.len().median()).reindex(LABELS)
tab["pool_n"] = pool.groupby("model").size().reindex(LABELS)
print(tab.to_string())
tw = train.text.str.split().str.len()
print("train 길이 분위수 5/25/50/75/95:", np.percentile(tw, [5, 25, 50, 75, 95]).tolist(),
      "| corpus:", np.percentile(corpus.ws_words, [5, 25, 50, 75, 95]).tolist())
print(f"corpus 중 train의 5~95% 길이 범위({np.percentile(tw, 5):.0f}~{np.percentile(tw, 95):.0f}단어)에 드는 비율: "
      f"{corpus.ws_words.between(np.percentile(tw, 5), np.percentile(tw, 95)).mean():.3f}")

head("4. 출처와 도메인")
print(corpus.groupby("source").size().sort_values(ascending=False).to_string())
print("\n라벨 x 출처(상위 8개 출처):")
top = corpus.source.value_counts().index[:8]
print(pd.crosstab(corpus.model, corpus.source.where(corpus.source.isin(top), "기타")).reindex(LABELS).to_string())
dom = corpus.domain.fillna("없음").str.split("/").str[0]
print("\n도메인(앞부분):", dom.value_counts().head(15).to_dict())
print("품질 플래그 비율:", {c: round(float(corpus[c].mean()), 3) for c in
                     ["truncation_metadata_missing", "license_unknown", "output_terms_review", "prompt_exact"]})
print("프롬프트 그룹당 라벨 수 분포:", corpus.groupby("prompt_group_id").model.nunique().value_counts().sort_index().to_dict())

head("5. 장르 표지 (비율)")


def genre(df):
    t = df.text
    return pd.Series({
        "n": len(df),
        "줄바꿈 있음": t.str.contains("\n").mean(),
        "문단 2개 이상": t.str.contains(r"\n\s*\n").mean(),
        "마크다운/목록/코드": t.map(lambda s: bool(MARKDOWN_RE.search(s))).mean(),
        "코드펜스": t.str.contains("```").mean(),
        "거절로 시작": t.map(lambda s: bool(REFUSAL_RE.match(s))).mean(),
        "1·2인칭(I/you) 있음": t.str.contains(r"\b(?:I|you|You|I'm|I’m)\b").mean(),
    })


g = pd.DataFrame({"train": genre(train), "pool(전체)": genre(pool), "corpus": genre(corpus)}).T
print(g.to_string())
print("\n출처별:")
print(corpus.groupby("source").apply(genre, include_groups=False)[lambda d: d.n >= 100].sort_values("n", ascending=False).to_string())
print("\n라벨별(corpus):")
print(corpus.groupby("model").apply(genre, include_groups=False).reindex(LABELS).to_string())

# train과 닮은 장르 조건: 한 문단, 마크다운 없음, 거절 아님
corpus["plain"] = (~corpus.text.str.contains("\n") & ~corpus.text.map(lambda s: bool(MARKDOWN_RE.search(s)))
                   & ~corpus.text.map(lambda s: bool(REFUSAL_RE.match(s))))
print(f"\n'한 문단 + 마크다운 없음 + 거절 아님' 셀: {int(corpus.plain.sum()):,}개")
print(pd.crosstab(corpus.model, corpus.plain).reindex(LABELS).T.to_string())

head("6. 표본 텍스트 (라벨마다 3개, 앞 300자)")
if "--samples" in sys.argv:
    for name in LABELS:
        for _, r in corpus[corpus.model == name].sample(3, random_state=1).iterrows():
            print(f"[{name} | {r.source} | {r.ws_words}단어] {r.text[:300]!r}")
    print("\ntrain 표본:")
    for _, r in train.sample(4, random_state=1).iterrows():
        print(f"[{r.model}] {r.text[:300]!r}")
else:
    print("(--samples를 줄 때만 출력합니다. 라이선스가 불분명한 외부 텍스트와 대회 지문을 저장소 로그에 남기지 않기 위해서입니다.)")

head("7. 문체 피처 27개: train / Open Pool 표본(노트북과 같은 45,000개) / corpus")
ng_pool = sample_pool(pool, 2500)
S = {"train": np.expm1(stylo_raw(train.text)), "pool": np.expm1(stylo_raw(ng_pool.text)),
     "corpus": np.expm1(stylo_raw(corpus.text))}
lab = {"train": train.model.values, "pool": ng_pool.model.values, "corpus": corpus.model.values}
np.savez_compressed(ROOT / "out" / "corpus" / "stylo_corpus.npz", S=np.log1p(S["corpus"]), model=corpus.model.values,
                    cell_id=corpus.cell_id.values)
overall = pd.DataFrame({k: v.mean(0) for k, v in S.items()}, index=STYLO_NAMES)
sd = S["train"].std(0) + 1e-9
overall["(pool-train)/sd"] = (S["pool"].mean(0) - S["train"].mean(0)) / sd
overall["(corpus-train)/sd"] = (S["corpus"].mean(0) - S["train"].mean(0)) / sd
print(overall.to_string())


def label_means(k):
    return pd.DataFrame(S[k], columns=STYLO_NAMES).groupby(lab[k]).mean().reindex(LABELS)


M = {k: label_means(k) for k in S}
print("\n라벨별 평균의 train과의 거리 (피처 27개에 대해 |차이|/train 표준편차의 평균):")
dist = pd.DataFrame({k: ((M[k] - M["train"]).abs() / sd).mean(1) for k in ("pool", "corpus")})
dist["corpus_n"] = corpus.groupby("model").size().reindex(LABELS)
print(dist.to_string())
print("\n라벨 간 차이가 보존되는가: 피처마다 라벨 18개 평균의 Spearman 상관 (train 대 pool, train 대 corpus)")
keys = ["em_dash", "spaced_em_dash", "en_dash", "semicolon", "colon", "comma", "paren", "curly_apos", "straight_apos",
        "curly_quote", "straight_quote", "contraction", "first_person", "second_person", "not_but", "sent_len_mean",
        "word_len_mean", "mattr", "func_ratio", "n_words"]
rows = []
for f in keys:
    rows.append((f, spearmanr(M["train"][f], M["pool"][f])[0], spearmanr(M["train"][f], M["corpus"][f])[0]))
sp = pd.DataFrame(rows, columns=["feature", "train~pool", "train~corpus"]).set_index("feature")
print(sp.to_string()); print("평균:", sp.mean().round(3).to_dict())
print("\n타이포그래피 표지의 라벨별 평균(100단어당): train | pool | corpus")
for f in ["em_dash", "semicolon", "curly_apos", "straight_apos", "contraction", "first_person"]:
    print(f"-- {f}")
    print(pd.DataFrame({k: M[k][f] for k in S}).T.round(2).to_string())

head("8. 도메인 분류기: 문체 27개로 train과 구별되는 정도 (AUC, 0.5면 구별 불가)")
rng = np.random.default_rng(0)


def domain_auc(k):
    idx = rng.choice(len(S[k]), min(len(S[k]), 8000), replace=False)
    X = np.log1p(np.vstack([S["train"], S[k][idx]]))
    t = np.r_[np.zeros(len(S["train"])), np.ones(len(idx))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    p = cross_val_predict(clf, X, t, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(t, p), p[len(S["train"]):], idx


for k in ("pool", "corpus"):
    auc, p, idx = domain_auc(k)
    print(f"train 대 {k}: AUC {auc:.3f} | {k} 행 중 'train일 확률' 0.5 이상인 비율 {np.mean(p < 0.5):.3f}")
# 길이 두 개(n_words, n_sents)를 빼고도 구별되는지
keep = [i for i, n in enumerate(STYLO_NAMES) if n not in ("n_words", "n_sents")]
for k in ("pool", "corpus"):
    idx = rng.choice(len(S[k]), min(len(S[k]), 8000), replace=False)
    X = np.log1p(np.vstack([S["train"], S[k][idx]]))[:, keep]
    t = np.r_[np.zeros(len(S["train"])), np.ones(len(idx))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    p = cross_val_predict(clf, X, t, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print(f"train 대 {k} (길이 피처 제외): AUC {roc_auc_score(t, p):.3f}")

head("9. 전이 점검: 한쪽 데이터만으로 학습해 train 2,699개 전체를 맞히는 BA (char_wb 2~4 TF-IDF + LinearSVC)")
y = train.model.map(L2I).values
COMPANY = np.array([["gpt", "gemini", "claude", "qwen", "glm", "kimi"].index(n.split("-")[0].rstrip("0123456789.")) for n in LABELS])


def transfer(texts, labels, name):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2, max_features=100_000,
                          lowercase=False)
    X = vec.fit_transform(texts)
    clf = LinearSVC(C=0.3, class_weight="balanced").fit(X, labels)
    pred = clf.predict(vec.transform(train.text))
    rec = pd.Series({LABELS[i]: float((pred[y == i] == i).mean()) for i in range(len(LABELS))})
    print(f"{name}: {len(texts):,}개로 학습 -> train BA {balanced_accuracy_score(y, pred):.4f} | "
          f"회사 6개 단위 BA {balanced_accuracy_score(COMPANY[y], COMPANY[pred]):.4f}", flush=True)
    return rec


counts = corpus.groupby("model").size().to_dict()
rec = pd.DataFrame({
    "corpus": transfer(corpus.text, corpus.model.map(L2I).values, "corpus 전체"),
    "pool(같은 개수)": transfer(*(lambda d: (d.text, d.model.map(L2I).values))(sample_pool(pool, counts, seed=1)),
                             "Open Pool에서 라벨별로 corpus와 같은 개수"),
    "corpus plain": transfer(corpus.text[corpus.plain], corpus.model.map(L2I).values[corpus.plain.values],
                             "corpus 중 한 문단·마크다운 없음·거절 아님"),
    "pool 45k": transfer(ng_pool.text, ng_pool.model.map(L2I).values, "Open Pool 표본 45,000개(노트북과 같은 표본)"),
})
rec["corpus_n"] = pd.Series(counts)
print("라벨별 recall:"); print(rec.reindex(LABELS).to_string())

# corpus 안에서는 라벨이 얼마나 잘 갈리는가(프롬프트 그룹 단위 5-fold). 이 값이 높고 전이가 낮으면 출처 냄새를 배운 것입니다.
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2, max_features=100_000, lowercase=False)
Xc = vec.fit_transform(corpus.text); yc = corpus.model.map(L2I).values
pred = np.zeros(len(yc), dtype=int)
for a, b in GroupKFold(5).split(Xc, yc, corpus.prompt_group_id):
    pred[b] = LinearSVC(C=0.3, class_weight="balanced").fit(Xc[a], yc[a]).predict(Xc[b])
print(f"corpus 안 프롬프트 그룹 5-fold BA: {balanced_accuracy_score(yc, pred):.4f}")
