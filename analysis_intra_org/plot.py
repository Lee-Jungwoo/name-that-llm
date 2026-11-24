"""out/*.csv로 그림 두 장을 그린다. 사용법: python plot.py  (paired.py, separability.py를 먼저 실행)"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
V4, NG, ST = "#0b0b0b", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK JP", "DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 9.5,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": INK, "axes.unicode_minus": False})
short = lambda m: m.replace("gemini-", "gem-").replace("claude-", "cl-").replace("-preview", "")


def separability():
    S = pd.read_csv(OUT / "separability.csv").sort_values("auc_v4", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 8.2))
    yy = np.arange(len(S))
    for i, r in S.iterrows():
        lo, hi = min(r.auc_v4, r.auc_ngram, r.auc_stylo), max(r.auc_v4, r.auc_ngram, r.auc_stylo)
        ax.plot([lo, hi], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(S.auc_stylo, yy, s=30, color=ST, label="전용 이진 분류기 · 문체 피처 27개", zorder=2)
    ax.scatter(S.auc_ngram, yy, s=30, color=NG, label="전용 이진 분류기 · 문자/단어 n-gram", zorder=2)
    ax.scatter(S.auc_v4, yy, s=42, color=V4, marker="D", label="v4 18-class (lp[A] − lp[B])", zorder=3)
    for i, r in S.iterrows():
        ax.text(1.003, i, f"{int(r.v4_cross_errors)}", va="center", fontsize=8.5, color=MUTED)
    ax.text(1.003, -1.1, "v4 상호\n오답 수", fontsize=8, color=MUTED, va="bottom")
    ax.set_yticks(yy, [f"{short(r.a)} · {short(r.b)}" + ("  *" if r.org == "cross" else "") for r in S.itertuples()])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color=MUTED, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("쌍별 구분 가능성 (AUC)", loc="left", fontsize=12.5, pad=30)
    ax.text(0, 1.035, "v4: 5-fold OOF(쌍마다 약 300개) · 전용 분류기: Open Pool로 학습, train 약 290개로 평가 · * 다른 기업 쌍",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "separability.png", dpi=200)


FEATS = ["word_len_mean", "long_word_ratio", "func_ratio", "mattr", "sent_len_mean", "sent_len_std", "n_sents",
         "comma", "semicolon", "colon", "em_dash", "spaced_em_dash", "contraction", "curly_apos", "curly_dq",
         "first_person", "second_person", "ly_adverb"]
KO = {"word_len_mean": "단어 길이", "long_word_ratio": "긴 단어 비율", "func_ratio": "기능어 비율", "mattr": "어휘 다양도",
      "sent_len_mean": "문장 길이", "sent_len_std": "문장 길이 편차", "n_sents": "문장 수", "comma": ",", "semicolon": ";",
      "colon": ":", "em_dash": "—", "spaced_em_dash": "띄어 쓴 —", "contraction": "축약형", "curly_apos": "’",
      "curly_dq": "“”", "first_person": "1인칭", "second_person": "2인칭", "ly_adverb": "-ly 부사"}


def heatmap():
    R = pd.read_csv(OUT / "paired.csv")
    R["key"] = R.a.map(short) + " − " + R.b.map(short) + np.where(R.org == "cross", "  *", "")
    keys = list(dict.fromkeys(R.key))
    M = R.pivot(index="key", columns="feature", values="dz").loc[keys, FEATS]
    sig = R.assign(s=(R.q < 0.05) & R.replicated).pivot(index="key", columns="feature", values="s").loc[keys, FEATS]
    show = M.where(sig.astype(bool))
    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    im = ax.imshow(show.values, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = show.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}".replace("+", "") if abs(v) >= 0.2 else "", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(v) > 0.7 else INK)
    ax.set_xticks(range(len(FEATS)), [KO[f] for f in FEATS], rotation=45, ha="right")
    ax.set_yticks(range(len(keys)), keys)
    orgs = R.drop_duplicates("key").org.values
    for i in range(1, len(keys)):
        if orgs[i] != orgs[i - 1]:
            ax.axhline(i - 0.5, color=INK, lw=1.2)
    nsig = sig.sum(1).values
    for i, n in enumerate(nsig):
        ax.text(len(FEATS) - 0.3, i, f"{int(n)}", va="center", fontsize=8.5, color=MUTED)
    ax.text(len(FEATS) - 0.3, -0.9, "유의 피처\n(27개 중)", fontsize=8, color=MUTED, va="bottom")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.07)
    cb.set_label("dz (+: 왼쪽 모델이 큼)")
    cb.outline.set_visible(False)
    ax.set_title("같은 프롬프트 짝 비교: 문체 피처 효과 크기", loc="left", fontsize=12.5, pad=30)
    ax.text(0, 1.025, "train 짝 137~147개 · 빈칸 = 유의하지 않음(BH q ≥ 0.05) 또는 Open Pool에서 방향이 반대 · * 다른 기업 쌍",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "paired_heatmap.png", dpi=200)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    separability()
    heatmap()
