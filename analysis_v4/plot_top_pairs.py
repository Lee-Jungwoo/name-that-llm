"""가장 많이 헷갈린 (정답, 예측) 쌍 Top 12 그림. 사용법: python plot_top_pairs.py figures/  (v4 fold 0, 540개)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load
from matplotlib.patches import Patch

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SAME, OTHER = "#2a78d6", "#eb6834"
TOP = 12
ORGS = ["gpt", "gemini", "claude", "qwen", "glm", "kimi"]

plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 9.5,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": INK})

outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
outdir.mkdir(parents=True, exist_ok=True)
y, p, _, _, labels = load(4)  # v4, fold 0
K = len(labels)
org = [next(o for o in ORGS if l.startswith(o)) for l in labels]

cm = np.zeros((K, K))
for k in range(K):
    cm[k] = np.bincount(p[y == k], minlength=K) / (y == k).sum()
np.fill_diagonal(cm, -1)
flat = np.argsort(-cm, axis=None, kind="stable")[:TOP]
pairs = [divmod(int(i), K) for i in flat]

fig, ax = plt.subplots(figsize=(8.4, 5.6))
rows = pairs[::-1]
vals = [cm[t, q] for t, q in rows]
colors = [SAME if org[t] == org[q] else OTHER for t, q in rows]
ax.barh(range(TOP), vals, height=0.62, color=colors)
for i, v in enumerate(vals):
    ax.text(v + 0.006, i, f"{v:.2f}", va="center", fontsize=9, color=INK)
ax.set_yticks(range(TOP), [f"{labels[t]} → {labels[q]}" for t, q in rows])
ax.set_xlim(0, 0.42)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(handles=[Patch(color=SAME, label="같은 기업"), Patch(color=OTHER, label="다른 기업")],
          loc="lower right", frameon=False, fontsize=9.5)
ax.set_title("v4 · 가장 많이 헷갈린 쌍 Top 12", loc="left", fontsize=12.5, pad=24)
ax.text(0, 1.02, "fold 0 지문 540개 · 정답 → 예측, 값 = 정답 지문 중 그 모델로 예측한 비율", transform=ax.transAxes,
        fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(outdir / "top_pairs_v4.png", dpi=200)
for t, q in pairs:
    print(f"{labels[t]} -> {labels[q]} {cm[t, q]:.3f} {'같은' if org[t] == org[q] else '다른'}")
