"""기업 내부 혼동 행렬 그림. 사용법: python plot_confusion.py figures/  (v4 fold 0, 540개)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUES = LinearSegmentedColormap.from_list(
    "seq", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORGS = [("GPT", "gpt"), ("Gemini", "gemini"), ("Claude", "claude"), ("GLM", "glm")]

plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 10,
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": INK, "ytick.color": INK})


def plot(y, p, labels, org, prefix, out):
    idx = [i for i, l in enumerate(labels) if l.startswith(prefix)]
    n = len(idx)
    cm = np.zeros((n, n + 1))
    for r, k in enumerate(idx):
        pred = p[y == k]
        for c, j in enumerate(idx):
            cm[r, c] = (pred == j).mean()
        cm[r, n] = 1 - cm[r, :n].sum()

    fig, ax = plt.subplots(figsize=(0.9 * (n + 1) + 2.2, 0.9 * n + 1.7))
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1, aspect="equal")
    ax.axvline(n - 0.5, color=SURFACE, lw=4)  # 외부 열을 따로 떼어 보이게
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=10,
                fontweight="bold" if r == c else "normal", color=SURFACE if v > 0.5 else INK)
    for d in range(n):
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False, ec=INK, lw=1.5))
    short = [labels[k].split("-", 1)[1] if "-" in labels[k] else labels[k] for k in idx]
    ax.set_xticks(range(n + 1), short + ["외부"], rotation=30, ha="right")
    ax.set_yticks(range(n), short)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"v4 · {org} 내부 혼동 행렬", loc="left", fontsize=12, pad=24)
    ax.text(0, 1.02, "fold 0 지문 540개 · 행 = 정답, 열 = 예측, 값 = 행 비율", transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
    outdir.mkdir(parents=True, exist_ok=True)
    y, p, _, _, labels = load(4)  # v4, fold 0
    for org, prefix in ORGS:
        plot(y, p, labels, org, prefix, outdir / f"cm_{org.lower()}.png")
        print("저장:", outdir / f"cm_{org.lower()}.png")
