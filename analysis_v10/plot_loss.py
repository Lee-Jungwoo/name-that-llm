"""v10 DeBERTa 학습 곡선: out/v10/deb_curve.csv(학습 중 기록) -> analysis_v10/loss_v10.png (CPU만, 가벼움).

deb_curve.csv 열: stage,step,n_steps,train_loss,val_loss,val_ba,minutes,skipped,overflow
  - train_loss: 학습 손실의 지수이동평균(0.98/0.02, label smoothing 0.05, Stage 1은 라벨 균형 가중치 포함)
  - val_loss / val_ba: Stage 1은 train 2,699개 전체, Stage 2 fold는 그 fold의 검증 지문. label smoothing·가중치 없는 CE(온도 1)
  - 이어 하기로 같은 (stage, step)이 두 번 있으면 나중 줄을 씁니다.
왼쪽: Stage 1 train / val loss. 가운데: Stage 2 fold별 val loss(실선)와 train loss(점선), full은 train loss만. 오른쪽: val BA.
사용: .venv/bin/python analysis_v10/plot_loss.py [csv 경로]
"""
import csv, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out/v10/deb_curve.csv"
OUT_PNG = ROOT / "analysis_v10/loss_v10.png"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # 범주 색 고정 순서 (fold0~4, full)
INK, MUTED = "#2b2b29", "#8a8984"


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        if name in have:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def read(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["stage"], int(r["step"]))] = r
    out = {}
    for (st, step), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        d = out.setdefault(st, {"n": int(r["n_steps"]), "step": [], "train": [], "vstep": [], "val": [], "ba": []})
        if step > 0 and r["train_loss"]:
            d["step"].append(step); d["train"].append(float(r["train_loss"]))
        if r["val_loss"]:
            d["vstep"].append(step); d["val"].append(float(r["val_loss"])); d["ba"].append(float(r["val_ba"]))
    return out


def style(ax, title, xlabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED)


def main():
    ko = pick_font()
    T = (lambda k, e: k) if ko else (lambda k, e: e)
    d = read(CSV)
    s1 = d.get("deb-stage1")
    s2 = {k: v for k, v in d.items() if k.startswith("deb-stage2")}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    if s1:
        ax.plot(s1["step"], s1["train"], color=SERIES[0], lw=2, label=T("train loss (이동평균, 가중)", "train loss (EMA, weighted)"))
        ax.plot(s1["vstep"], s1["val"], "o-", color=SERIES[1], lw=2, ms=5, label=T("val loss (train 2,699개)", "val loss (train 2,699)"))
        ax.set_xlim(0, s1["n"])
    style(ax, "Stage 1 (Open Pool 230,950" + T("행)", " rows)"), T("스텝", "step"))
    ax.set_ylabel("loss", color=MUTED)
    ax.legend(frameon=False)
    ax = axes[1]
    for i, name in enumerate(sorted(s2, key=lambda n: (n.endswith("full"), n))):
        v, c, lab = s2[name], SERIES[i % len(SERIES)], name.replace("deb-stage2-", "")
        if v["val"]:
            ax.plot(v["vstep"], v["val"], "o-", color=c, lw=2, ms=4, label=f"{lab} val")
        ax.plot(v["step"], v["train"], "--", color=c, lw=1.2, label=f"{lab} train")
    style(ax, T("Stage 2 (fold별, 2 epoch)", "Stage 2 (per fold, 2 epochs)"), T("스텝", "step"))
    ax.set_ylabel("loss", color=MUTED)
    if s2:
        ax.legend(frameon=False, ncol=2, fontsize=8)
    ax = axes[2]
    if s1:
        ax.plot([x / s1["n"] for x in s1["vstep"]], s1["ba"], "o-", color=SERIES[0], lw=2, ms=4, label="Stage 1 (train 2,699)")
    for i, name in enumerate(sorted(k for k in s2 if not k.endswith("full"))):
        v = s2[name]
        if v["ba"]:
            ax.plot([x / v["n"] for x in v["vstep"]], v["ba"], "o-", color=SERIES[(i + 1) % len(SERIES)], lw=1.5, ms=3,
                    label=name.replace("deb-stage2-", "Stage 2 "))
    style(ax, "val balanced accuracy", T("진행률 (단계 안 스텝 / 전체)", "progress within stage"))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, facecolor="white")
    msg = f"저장: {OUT_PNG}"
    if s1:
        msg += f" | stage1 train {len(s1['step'])}점, val {len(s1['val'])}점 (마지막 val loss {s1['val'][-1]:.4f}, BA {s1['ba'][-1]:.4f})" if s1["val"] else ""
    print(msg + f" | stage2 {sorted(s2)}")


if __name__ == "__main__":
    main()
