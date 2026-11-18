"""report.md의 점수·recall·기업 단위 그래프. 사용법: python plot_tables.py . figures/
모든 버전을 fold 0 지문 540개로 잰다(common.py)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import VERSIONS, load, org_of as org_names, recall
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # 범주형 1~4번
VERSION = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # v1~v4, 순서형 파랑
BLUES = LinearSegmentedColormap.from_list(
    "seq", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORGS = [("GPT", "gpt"), ("Gemini", "gemini"), ("Claude", "claude"), ("Qwen", "qwen"), ("GLM", "glm"), ("Kimi", "kimi")]
VERS = ["v1", "v2", "v3", "v4"]

KAGGLE = [0.451, 0.507, 0.536, 0.545]   # 캐글 퍼블릭 점수 (v3는 Sinkhorn 켠 제출, v4는 full 제출)

plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 9.5,
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": INK, "ytick.color": MUTED})


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def titles(ax, title, sub):
    ax.set_title(title, loc="left", fontsize=12.5, pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=MUTED)


def grouped_bars(ax, groups, series, colors, width, fmt="{:.3f}", label_size=8):
    n = len(series)
    x = np.arange(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        xs = x + (i - (n - 1) / 2) * width
        bars = ax.bar(xs, [0 if v is None else v for v in vals], width * 0.9, color=colors[i], label=name)
        for xi, v in zip(xs, vals):
            if v is not None:
                ax.text(xi, v + 0.012, fmt.format(v), ha="center", va="bottom", fontsize=label_size, color=INK)
    ax.set_xticks(x, groups)
    return x


def scores_from_data():
    """fold 0 기준 n-gram, LoRA, 블렌드 BA와 캐글 퍼블릭 점수."""
    out = {"n-gram": [], "LoRA": [], "블렌드": []}
    for v in VERSIONS:
        y, blend, ng, lora, _ = load(v)
        for name, p in (("n-gram", ng), ("LoRA", lora), ("블렌드", blend)):
            out[name].append(recall(y, p).mean())
    out["캐글 퍼블릭"] = KAGGLE
    return out


def heat(ax, M, rows, cols, best_mask):
    ax.imshow(M, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                fontweight="bold" if best_mask[r, c] else "normal", color=SURFACE if v > 0.55 else INK)
        if best_mask[r, c]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
    ax.set_xticks(range(len(cols)), cols)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)


def main(root, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    data = {v: load(v) for v in VERSIONS}
    preds = {v: (d[0], d[1]) for v, d in data.items()}
    labels = data[4][4]
    K = len(labels)
    org_of = org_names(labels)
    SCORES = scores_from_data()
    saved = []

    # 1. 전체 점수: validation vs 캐글 퍼블릭
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    grouped_bars(ax, VERS, {"validation 점수": SCORES["블렌드"], "캐글 퍼블릭 점수": SCORES["캐글 퍼블릭"]},
                 CAT[:2], 0.36, fmt="{:.3f}", label_size=8.5)
    ax.set_ylim(0, 0.72)
    style(ax)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    titles(ax, "버전별 validation 점수", "balanced accuracy · validation은 네 버전 모두 fold 0 지문 540개")
    fig.tight_layout()
    fig.savefig(outdir / "scores.png", dpi=200); plt.close(fig); saved.append("scores.png")

    # 2. 모델별 recall v1~v4 (v4 순서)
    R = np.stack([recall(*preds[v], K) for v in range(1, 5)], 1)
    order = np.argsort(-R[:, 3], kind="stable")
    M = R[order]
    best = np.isclose(M, M.max(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(6.4, 8.2))
    heat(ax, M, [f"{labels[k]} ({org_of[k]})" for k in order], VERS, best)
    ax.set_title("버전별 모델 validation recall", loc="left", fontsize=12.5, pad=36)
    ax.text(0, 1.045, "fold 0 지문 540개(모델당 30개) · v4 순서 · 테두리 = 네 버전 중 최고",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(outdir / "recall_versions.png", dpi=200); plt.close(fig); saved.append("recall_versions.png")

    # 3·4. 기업 단위
    org_acc, within = {}, {}
    for v in range(1, 5):
        y, p = preds[v]
        yo, po = np.array(org_of)[y], np.array(org_of)[p]
        for n, _ in ORGS:
            m = yo == n
            org_acc[(v, n)] = (po[m] == n).mean()
            hit = m & (po == n)
            within[(v, n)] = (p[hit] == y[hit]).mean()
        org_acc[(v, "전체")] = (yo == po).mean()

    names = [n for n, _ in ORGS] + ["전체"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    grouped_bars(ax, names, {f"v{v}": [org_acc[(v, n)] for n in names] for v in range(1, 5)},
                 VERSION, 0.2, fmt="{:.2f}", label_size=7.5)
    ax.set_ylim(0, 1.12)
    style(ax)
    ax.legend(loc="upper right", frameon=False, ncol=4, bbox_to_anchor=(1, 1.13))
    titles(ax, "기업을 맞힌 비율", "fold 0 지문 540개")
    fig.tight_layout()
    fig.savefig(outdir / "org_accuracy.png", dpi=200); plt.close(fig); saved.append("org_accuracy.png")

    multi = [(n, sum(o == n for o in org_of)) for n, _ in ORGS if sum(o == n for o in org_of) > 1]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = grouped_bars(ax, [f"{n} ({c}개)" for n, c in multi],
                     {f"v{v}": [within[(v, n)] for n, _ in multi] for v in range(1, 5)},
                     VERSION, 0.2, fmt="{:.2f}", label_size=7.5)
    for xi, (_, c) in zip(x, multi):
        ax.plot([xi - 0.44, xi + 0.44], [1 / c, 1 / c], color=INK, lw=1.4, ls=(0, (3, 2)))
    ax.plot([], [], color=INK, lw=1.4, ls=(0, (3, 2)), label="기업 안에서 찍을 때")
    ax.set_ylim(0, 0.95)
    style(ax)
    ax.legend(loc="upper left", frameon=False, ncol=5, fontsize=8.5)
    titles(ax, "기업을 맞혔을 때 모델까지 맞힌 비율", "점선 = 기업 안에서 무작위로 고를 때 · fold 0 지문 540개")
    fig.tight_layout()
    fig.savefig(outdir / "org_within.png", dpi=200); plt.close(fig); saved.append("org_within.png")

    # 부록 1. 버전별 모델 점수
    fig, ax = plt.subplots(figsize=(9, 4.4))
    grouped_bars(ax, VERS, SCORES, CAT, 0.2, fmt="{:.3f}", label_size=7)
    ax.set_ylim(0, 0.72)
    style(ax)
    ax.legend(loc="upper left", frameon=False, ncol=4)
    titles(ax, "부록 · 버전별 모델 점수", "n-gram·LoRA·블렌드는 fold 0 지문 540개 · 캐글은 퍼블릭 점수")
    fig.tight_layout()
    fig.savefig(outdir / "scores_by_model.png", dpi=200); plt.close(fig); saved.append("scores_by_model.png")

    # 부록 2. v4 방식별 recall
    y, blend4, ng4, lora4, _ = data[4]
    R3 = np.stack([recall(y, ng4, K), recall(y, lora4, K), recall(y, blend4, K)], 1)
    order = np.argsort(-R3[:, 2], kind="stable")
    M = R3[order]
    best = np.isclose(M, M.max(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(5.8, 8.2))
    heat(ax, M, [f"{labels[k]} ({org_of[k]})" for k in order], ["n-gram", "LoRA", "블렌드"], best)
    ax.set_title("부록 · v4 방식별 모델 recall", loc="left", fontsize=12.5, pad=36)
    ax.text(0, 1.045, "fold 0 지문 540개 · 블렌드 순서 · 테두리 = 세 방식 중 최고", transform=ax.transAxes,
            fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(outdir / "recall_methods_v4.png", dpi=200); plt.close(fig); saved.append("recall_methods_v4.png")

    for f in saved:
        print("저장:", outdir / f)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
