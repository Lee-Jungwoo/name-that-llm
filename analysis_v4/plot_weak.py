"""약한 모델의 예측 행선지 상위 4개 그림. 사용법: python plot_weak.py figures/  (v4 fold 0, 540개)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CORRECT, WRONG = "#2a78d6", "#b9b8b3"
N_WEAK = 7  # v4 fold 0 recall 하위 7개 (7위와 동점이면 함께 넣음)
TOP = 4

plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 9.5,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": INK})

outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
outdir.mkdir(parents=True, exist_ok=True)
y, p, _, _, labels = load(4)  # v4, fold 0
_rec = np.array([(p[y == k] == k).mean() for k in range(len(labels))])
_cut = np.sort(_rec)[N_WEAK - 1]
WEAK = [labels[k] for k in np.argsort(_rec, kind="stable") if _rec[k] <= _cut + 1e-9]

fig, axes = plt.subplots((len(WEAK) + 1) // 2, 2, figsize=(10, 2.15 * ((len(WEAK) + 1) // 2) + 0.9))
for ax, name in zip(axes.flat, WEAK):
    k = labels.index(name)
    share = np.bincount(p[y == k], minlength=len(labels)) / (y == k).sum()
    top = np.argsort(-share, kind="stable")[:TOP][::-1]
    vals = share[top]
    ax.barh(range(TOP), vals, height=0.62, color=[CORRECT if j == k else WRONG for j in top])
    for i, (j, v) in enumerate(zip(top, vals)):
        ax.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=9,
                color=INK, fontweight="bold" if j == k else "normal")
    ax.set_yticks(range(TOP), [labels[j] for j in top])
    ax.set_xlim(0, 0.72)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"정답 {name}", loc="left", fontsize=10.5)

for ax in axes.flat[len(WEAK):]:
    ax.axis("off")

fig.suptitle("v4 · 약한 모델은 어디로 틀리나", x=0.01, y=0.995, ha="left", fontsize=12.5)
fig.text(0.01, 0.955, f"fold 0 지문 540개 · recall 하위 {len(WEAK)}개 · 예측 비율 상위 4개 · 파란 막대 = 정답으로 예측 (= recall), 회색 = 다른 모델로 예측",
         ha="left", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(outdir / "weak_models_v4.png", dpi=200)
print("저장:", outdir / "weak_models_v4.png")
