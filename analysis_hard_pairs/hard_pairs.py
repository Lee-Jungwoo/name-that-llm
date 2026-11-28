"""v8(= v9의 LoRA 외 부분) 게이트 뒤 OOF에서 가장 많이 헷갈리는 모델 쌍을 골라, 두 모델의 지문이 무엇이 다른지
설명 가능한 단위(문체 피처, 표지 토큰)로 정리합니다. CPU만 씁니다.

입력
  - data/train.csv (지문 2,699개), analysis_v8/oof_v8.npz (세 분기 OOF 로짓), analysis_v4/oof_v4.npz (fold)
  - Open Pool: ntl_open_pool.parquet (경로를 첫 번째 인자로). flag_kb_proxy_after_ruling 행은 뺍니다.
방법
  1. 예측: analysis_v8/compare_v7.py와 같은 블렌드(lora 0.6 / deb 0.2 / ng 0.2) + 회사 게이트.
  2. 쌍 고르기: 상호 오답 수 (A→B) + (B→A)가 큰 순서.
  3. 문체 27개(analysis_intra_org/features.py): 같은 프롬프트 짝의 차이로 효과 크기 dz = 평균(A−B) / 표준편차(A−B).
     train 짝(프롬프트 150개)과 풀 짝(같은 prompt_canon) 둘 다 계산합니다.
  4. 표지 토큰: 단어 1·2-gram, 구두점, 첫 단어, 줄바꿈 등. 풀 짝 지문에서 '나온 지문 비율'(DF)을 비교하고
     McNemar 형 짝 z = (b − c) / √(b + c) 로 순위를 매깁니다(b = A에만 나온 짝, c = B에만 나온 짝).
     train 짝에서도 같은 방향인지 확인합니다.
  5. 설명 가능한 분류기: 문체 27개 + 표지 토큰 상위 60개(존재 여부)로 풀 짝에서 로지스틱 회귀를 학습하고
     train의 A·B 지문으로 AUC를 잽니다. v8의 쌍 점수 lp[A] − lp[B]와의 Spearman 상관도 봅니다.

  6. 틀린 지문: 쌍마다 v8이 맞힌 지문과 상대 모델로 틀린 지문의 단어 수, 하이픈 복합어, 핵심 표지 비율을 비교합니다.

사용: python analysis_hard_pairs/hard_pairs.py <ntl_open_pool.parquet> [OOF npz] > analysis_hard_pairs/logs/hard_pairs.log
      OOF npz는 기본값 analysis_v8/oof_v8.npz. 같은 키(ng_logits, lora_logits, deb_logits, y, ids)의 v9 OOF를 넣으면 v9 기준으로 다시 냅니다.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bas, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis_intra_org"))
from features import FEATURES, feats  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
          "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
          "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
          "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]
FAMILIES = {p: [i for i, l in enumerate(LABELS) if l.startswith(p)] for p in ("gpt", "gemini", "claude", "glm")}
GATE_ALPHA = dict(gpt=0.75, gemini=0.25, claude=0.5, glm=1.0)
FIXED = dict(lora=0.6, deb=0.2, ng=0.2)
GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3)
N_PAIRS = 8
N_TOK = 30  # 쪽마다 분류기에 넣는 표지 토큰 수

# ---------------------------------------------------------------- 1. v8 게이트 뒤 예측
v4 = np.load(ROOT / "analysis_v4/oof_v4.npz", allow_pickle=True)
y, fold = v4["y"], v4["fold"]
folds = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(5)]
OOF = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "analysis_v8/oof_v8.npz"
z = np.load(OOF, allow_pickle=True)
print("OOF:", OOF)
assert (z["y"] == y).all()


def lsm(x, t):
    x = x / t
    return x - logsumexp(x, axis=1, keepdims=True)


def best_t(s):
    return GRID[int(np.argmin([-lsm(s, t)[np.arange(len(y)), y].mean() for t in GRID]))]


L = {k: lsm(z[f"{k}_logits"], best_t(z[f"{k}_logits"])) for k in ("ng", "lora", "deb")}
blend = sum(FIXED[k] * L[k] for k in FIXED)
out = blend.copy()
for name, cols in FAMILIES.items():
    feat = np.hstack([L[k][:, cols] for k in ("ng", "lora", "deb")])
    spec = np.zeros((len(y), len(cols)))
    for a, b in folds:
        rows = a[np.isin(y[a], cols)]
        spec[b] = LogisticRegression(C=1.0, max_iter=3000).fit(
            feat[rows], np.searchsorted(cols, y[rows])).predict_log_proba(feat[b])
    sub = out[:, cols]
    mass = logsumexp(sub, axis=1, keepdims=True)
    new = log_softmax((1 - GATE_ALPHA[name]) * (sub - mass) + GATE_ALPHA[name] * spec, axis=1) + mass
    out[:, cols] = np.where(np.isin(out.argmax(1), cols)[:, None], new, sub)
pred = out.argmax(1)
print(f"OOF BA: 블렌드 {bas(y, blend.argmax(1)):.4f} -> 게이트 {bas(y, pred):.4f} (v8 커널 로그 0.5982 -> 0.6023)")

train = pd.read_csv(ROOT / "data/train.csv")
train = train.set_index("id").loc[z["ids"]].reset_index()
assert (train.model.map(LABELS.index).values == y).all()
train["pred"] = [LABELS[i] for i in pred]

# ---------------------------------------------------------------- 2. 쌍 고르기
C = np.zeros((18, 18), int)
np.add.at(C, (y, pred), 1)
pairs = sorted(((C[i, j] + C[j, i], i, j) for i in range(18) for j in range(i + 1, 18)), reverse=True)[:N_PAIRS]
print("\n== 상호 오답이 많은 쌍 (정답 150개씩, 게이트 뒤) ==")
print(f"{'A':24s} {'B':24s} A→B B→A 합   recall A  recall B")
for s, i, j in pairs:
    print(f"{LABELS[i]:24s} {LABELS[j]:24s} {C[i, j]:3d} {C[j, i]:3d} {s:3d}   {C[i, i] / C[i].sum():.3f}    {C[j, j] / C[j].sum():.3f}")

HYPH = re.compile(r"[A-Za-z]-[A-Za-z]")
WR = re.compile(r"[A-Za-z’']+")
train["nw"] = train.text.map(lambda t: len(WR.findall(t)))
train["hy"] = train.text.map(lambda t: bool(HYPH.search(t)))
train["ok"] = train.pred == train.model
print("\n== 지문 길이(단어 수) 구간별 정확도 (18개 모델 전체, 게이트 뒤 argmax) ==")
for b, d in train.groupby(pd.qcut(train.nw, 4), observed=True):
    print(f"  {d.nw.min():3d}~{d.nw.max():3d}단어: 정확도 {d.ok.mean():.3f} ({len(d)}개)")

# ---------------------------------------------------------------- 3. 풀
pool = pd.read_parquet(sys.argv[1], columns=["text", "model", "prompt_canon", "flag_kb_proxy_after_ruling", "gen_reasoning"])
pool = pool[(~pool.flag_kb_proxy_after_ruling.astype(str).eq("True")) & (pool.prompt_canon != "")]
pool = pool.drop_duplicates("text").reset_index(drop=True)
pool["nw"] = pool.text.map(lambda t: len(WR.findall(t)))
pool["hy"] = pool.text.map(lambda t: bool(HYPH.search(t)))
print(f"\n풀: {len(pool):,}개 (kb_proxy 판정 행, prompt_canon 없는 행, 중복 지문 제외)")

# ---------------------------------------------------------------- 4. 토큰
WORD = re.compile(r"[a-z]+(?:['’][a-z]+)?")
PUNCT = {"—": "—", " — ": "␣—␣", "–": "–", ";": ";", ":": ":", "(": "(", "!": "!", "?": "?", "…": "…",
         "“": "“", '"': '"', "’": "’", "'": "'", "*": "*", "\n": "\\n", "...": "...", "/": "/", "-": "-"}


def tokens(t):
    ws = WORD.findall(t.lower().replace("’", "'"))
    s = set(ws) | {f"{a} {b}" for a, b in zip(ws, ws[1:])}
    s |= {f"[{name}]" for p, name in PUNCT.items() if p in t}
    first = WORD.findall(t.lower())
    if first:
        s.add(f"[첫 단어] {first[0]}")
    last = t.rstrip()[-1:] if t.strip() else ""
    s.add(f"[끝 글자] {last}")
    return s


def paired_frame(df, key, a, b):
    """같은 key(프롬프트)에 A와 B가 모두 있는 행만, key마다 한 개씩 짝지은 (A 지문, B 지문)."""
    da = df[df.model == a].drop_duplicates(key).set_index(key).text
    db = df[df.model == b].drop_duplicates(key).set_index(key).text
    k = da.index.intersection(db.index)
    return da.loc[k].tolist(), db.loc[k].tolist()


def ff(texts):
    return pd.DataFrame([feats(t) for t in texts])


def dz(fa, fb):
    d = fa.values - fb.values
    sd = d.std(0, ddof=1)
    return np.where(sd > 0, d.mean(0) / np.where(sd > 0, sd, 1), 0.0)


def tok_stats(ta, tb, min_pairs=30):
    """짝마다 토큰 존재 여부 비교. 반환: {tok: (b, c, z)} (b = A에만, c = B에만)."""
    b, c = Counter(), Counter()
    for x, w in zip(ta, tb):
        sx, sw = tokens(x), tokens(w)
        b.update(sx - sw)
        c.update(sw - sx)
    res = {}
    for t in set(b) | set(c):
        n = b[t] + c[t]
        if n >= min_pairs:
            res[t] = (b[t], c[t], (b[t] - c[t]) / np.sqrt(n))
    return res


def df_rate(texts, tok):
    return np.mean([tok in tokens(t) for t in texts])


rows_pairs, rows_feat, rows_tok = [], [], []
for s, i, j in pairs:
    A, B = LABELS[i], LABELS[j]
    print(f"\n{'=' * 100}\n## {A}  vs  {B}   (v8 상호 오답 {C[i, j]} + {C[j, i]} = {s})")
    pa, pb = paired_frame(pool, "prompt_canon", A, B)
    ta, tb = paired_frame(train, "prompt_id", A, B)
    print(f"짝: 풀 {len(pa):,}개, train {len(ta)}개")

    # v8 쌍 점수
    m = train.model.isin([A, B]).values
    sc = out[m][:, i] - out[m][:, j]
    yy = (train.model[m] == A).values.astype(int)
    auc_v8 = roc_auc_score(yy, sc)
    # 같은 프롬프트 짝에서 v8이 A 지문 쪽 점수를 더 높게 준 비율
    tr = train[m].assign(sc=sc)
    pv = tr.pivot_table(index="prompt_id", columns="model", values="sc")
    pair_acc_v8 = float((pv[A] > pv[B]).mean())
    print(f"v8 쌍 AUC(lp[A] − lp[B]) {auc_v8:.3f}, 같은 프롬프트 짝 정확도 {pair_acc_v8:.3f}")

    # 문체 27개
    FA, FB, GA, GB = ff(pa), ff(pb), ff(ta), ff(tb)
    dz_pool, dz_train = dz(FA, FB), dz(GA, GB)
    order = np.argsort(-np.abs(dz_pool))
    print("\n문체 피처 (같은 프롬프트 짝, dz = 평균(A−B)/표준편차(A−B); |dz| 0.2 작음, 0.5 중간)")
    print(f"  {'피처':34s} {'풀 A':>7s} {'풀 B':>7s} {'풀 dz':>6s} {'train A':>8s} {'train B':>8s} {'train dz':>8s}")
    for k in order[:10]:
        f = FA.columns[k]
        same = "" if np.sign(dz_pool[k]) == np.sign(dz_train[k]) else "  (방향 다름)"
        print(f"  {FEATURES[f]:34s} {FA[f].mean():7.2f} {FB[f].mean():7.2f} {dz_pool[k]:+6.2f} "
              f"{GA[f].mean():8.2f} {GB[f].mean():8.2f} {dz_train[k]:+8.2f}{same}")
        rows_feat.append(dict(A=A, B=B, feature=f, desc=FEATURES[f], pool_A=FA[f].mean(), pool_B=FB[f].mean(),
                              dz_pool=dz_pool[k], train_A=GA[f].mean(), train_B=GB[f].mean(), dz_train=dz_train[k]))

    # 표지 토큰
    st = tok_stats(pa, pb)
    stt = tok_stats(ta, tb, min_pairs=1)
    ranked = sorted(st.items(), key=lambda kv: -abs(kv[1][2]))
    sideA = [t for t, v in ranked if v[2] > 0][:N_TOK]
    sideB = [t for t, v in ranked if v[2] < 0][:N_TOK]
    print("\n표지 토큰 (지문에 나온 비율 DF, 같은 프롬프트 짝. 짝 z = (A에만 − B에만)/√(합); train 칸은 같은 계산)")
    for side, toks in ((A, sideA), (B, sideB)):
        print(f"  [{side} 쪽]")
        print(f"    {'토큰':24s} {'풀 DF A':>8s} {'풀 DF B':>8s} {'짝 z':>6s} {'train DF A':>10s} {'train DF B':>10s} {'train 짝(A만/B만)':>16s}")
        for t in toks[:12]:
            bb, cc, zz = st[t]
            tb_, tc_, _ = stt.get(t, (0, 0, 0))
            dfa, dfb = df_rate(pa, t), df_rate(pb, t)
            gfa, gfb = df_rate(ta, t), df_rate(tb, t)
            flag = "" if (gfa - gfb) * zz > 0 else "  (train 방향 다름)"
            print(f"    {t:24s} {dfa:8.3f} {dfb:8.3f} {zz:+6.1f} {gfa:10.3f} {gfb:10.3f} {tb_:>8d}/{tc_:<6d}{flag}")
            rows_tok.append(dict(A=A, B=B, side=side, token=t, pool_df_A=dfa, pool_df_B=dfb, z_pool=zz,
                                 train_df_A=gfa, train_df_B=gfb, train_onlyA=tb_, train_onlyB=tc_))

    # 설명 가능한 분류기: 문체 27 + 표지 토큰 60
    vocab = sideA + sideB

    def X(texts, F):
        T = np.array([[t in tokens(x) for t in vocab] for x in texts], float)
        S = np.log1p(F.values.clip(0))
        return T, S

    TA, SA = X(pa, FA)
    TB, SB = X(pb, FB)
    mu, sd = np.vstack([SA, SB]).mean(0), np.vstack([SA, SB]).std(0) + 1e-9
    Xtr = np.vstack([np.hstack([TA, (SA - mu) / sd]), np.hstack([TB, (SB - mu) / sd])])
    ytr = np.r_[np.ones(len(TA)), np.zeros(len(TB))]
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(Xtr, ytr)
    Ftr_all = ff(train.text[m])
    Tt, St = X(train.text[m].tolist(), Ftr_all)
    score = clf.decision_function(np.hstack([Tt, (St - mu) / sd]))
    auc_x = roc_auc_score(yy, score)
    clf_t = LogisticRegression(C=0.1, max_iter=5000).fit(Xtr[:, :len(vocab)], ytr)
    auc_tok = roc_auc_score(yy, clf_t.decision_function(Tt))
    clf_s = LogisticRegression(C=0.1, max_iter=5000).fit(Xtr[:, len(vocab):], ytr)
    auc_sty = roc_auc_score(yy, clf_s.decision_function((St - mu) / sd))
    rho = spearmanr(score, sc).statistic
    print(f"\n설명 가능한 분류기(풀 짝으로 학습, train A·B 지문 {m.sum()}개로 평가): AUC 토큰 60개 {auc_tok:.3f}, "
          f"문체 27개 {auc_sty:.3f}, 합쳐서 {auc_x:.3f}  |  v8 {auc_v8:.3f}  |  v8 점수와의 Spearman ρ {rho:.2f}")

    # v8이 틀린 지문은 설명 가능한 점수로 보면 어디쯤 있나
    wrongA = (yy == 1) & (sc < 0)
    wrongB = (yy == 0) & (sc > 0)
    okA, okB = (yy == 1) & (sc >= 0), (yy == 0) & (sc <= 0)
    q = lambda mask: f"{np.median(score[mask]):+.2f}" if mask.any() else "  -"
    print(f"설명 가능한 점수의 중앙값(+ = {A} 쪽): {A} 맞힘 {q(okA)} / {A}인데 {B} 쪽 {q(wrongA)} | "
          f"{B} 맞힘 {q(okB)} / {B}인데 {A} 쪽 {q(wrongB)}")
    agree_wrong = np.mean(np.sign(score[wrongA | wrongB]) == np.sign(sc[wrongA | wrongB])) if (wrongA | wrongB).any() else np.nan
    print(f"v8이 이 쌍 안에서 순서를 거꾸로 매긴 지문 {int((wrongA | wrongB).sum())}개 중 설명 가능한 분류기도 같은 쪽으로 틀린 비율 {agree_wrong:.2f}")

    coef = pd.Series(clf.coef_[0], index=vocab + list(FA.columns)).sort_values()
    print("분류기 계수 상위(+ = A 쪽):", ", ".join(f"{k} {v:+.2f}" for k, v in coef[::-1][:6].items()))
    print("분류기 계수 하위(− = B 쪽):", ", ".join(f"{k} {v:+.2f}" for k, v in coef[:6].items()))

    rows_pairs.append(dict(A=A, B=B, a2b=C[i, j], b2a=C[j, i], recall_A=C[i, i] / C[i].sum(), recall_B=C[j, j] / C[j].sum(),
                           pool_pairs=len(pa), train_pairs=len(ta), auc_v8=auc_v8, pair_acc_v8=pair_acc_v8,
                           auc_tokens=auc_tok, auc_stylo=auc_sty, auc_explain=auc_x, spearman_v8=rho,
                           wrong_agree=agree_wrong))

    # 길이와 하이픈 복합어
    pvn = tr.assign(nw=train.nw[m].values).pivot_table(index="prompt_id", columns="model", values="nw")
    dn = (pvn[A] - pvn[B]).dropna()
    ppn = pool[pool.model.isin([A, B])].pivot_table(index="prompt_canon", columns="model", values="nw", aggfunc="first")
    dpn = (ppn[A] - ppn[B]).dropna()
    print(f"\n단어 수 평균: train {train.nw[train.model == A].mean():.0f} vs {train.nw[train.model == B].mean():.0f} "
          f"(짝 dz {dn.mean() / dn.std():+.2f}, A가 더 긴 짝 {np.mean(dn > 0):.2f}) | 풀 {pool.nw[pool.model == A].mean():.0f} vs "
          f"{pool.nw[pool.model == B].mean():.0f} (짝 dz {dpn.mean() / dpn.std():+.2f}) | v8 쌍 점수와의 Spearman ρ {spearmanr(sc, train.nw[m]).statistic:+.2f}")
    print(f"하이픈 복합어(글자-글자)가 나온 비율: train {train.hy[train.model == A].mean():.2f} vs {train.hy[train.model == B].mean():.2f} | "
          f"풀 {pool.hy[pool.model == A].mean():.2f} vs {pool.hy[pool.model == B].mean():.2f} | v8 쌍 점수와의 Spearman ρ {spearmanr(sc, train.hy[m]).statistic:+.2f}")

    # 틀린 지문 프로필: 맞힌 지문 / 상대 모델로 틀린 지문 / 상대 모델 전체
    keys = [t for t in sideA[:3]] + [t for t in sideB[:3]]
    print(f"틀린 지문 프로필 (토큰 칸은 나온 지문 비율, 토큰 {', '.join(keys)})")
    for X_, Y_ in ((A, B), (B, A)):
        okd = train[(train.model == X_) & (train.pred == X_)]
        wrd = train[(train.model == X_) & (train.pred == Y_)]
        oth = train[train.model == Y_]
        def prof(d):
            if not len(d):
                return "  -"
            toks = " ".join(f"{np.mean([k in tokens(t) for t in d.text]):.2f}" for k in keys)
            return f"{len(d):3d}개 | 단어 수 {d.nw.mean():5.1f} 단어 길이 {np.mean([feats(t)['word_len_mean'] for t in d.text]):.2f} 문장 길이 {np.mean([feats(t)['sent_len_mean'] for t in d.text]):5.1f} 하이픈 {d.hy.mean():.2f} | {toks}"
        print(f"  {X_} 맞힘        {prof(okd)}")
        print(f"  {X_} → {Y_} 틀림 {prof(wrd)}")
        print(f"  {Y_} 전체        {prof(oth)}")

    # 예: 같은 프롬프트에서 v8이 A를 B로 본 지문과 그 프롬프트의 B 지문
    ex = tr[(tr.model == A) & (tr.pred == B)]
    if len(ex):
        e = ex.iloc[0]
        other = train[(train.prompt_id == e.prompt_id) & (train.model == B)].iloc[0]
        print(f"\n예 (prompt {e.prompt_id}, v8은 {A} 지문을 {B}로 예측, 같은 프롬프트의 {B} 지문 예측은 {other.pred})")
        print(f"  [{A}] {e.text[:420]}")
        print(f"  [{B}] {other.text[:420]}")

pd.DataFrame(rows_pairs).to_csv(OUT / "pairs.csv", index=False)
pd.DataFrame(rows_feat).to_csv(OUT / "stylo.csv", index=False)
pd.DataFrame(rows_tok).to_csv(OUT / "tokens.csv", index=False)
print("\n저장:", OUT / "pairs.csv", OUT / "stylo.csv", OUT / "tokens.csv")
