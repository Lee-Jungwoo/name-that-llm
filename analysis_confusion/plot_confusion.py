"""회사(기업) 내부 혼동 행렬을 v4와 v7에 대해 같은 기준(OOF 2,699개)으로 다시 그립니다.

  v4 예측: analysis_v4/oof_v4.npz 의 eval_lp (n-gram + LoRA 블렌드, w = 0.6)
  v7 예측: analysis_v7/oof_v7.npz (제출 커널이 저장한 세 분기의 OOF 로짓)에 v7 노트북의
           5장(3분기 블렌드)과 5-1장(회사 게이트) 셀을 그대로 실행한 결과
둘 다 Sinkhorn을 적용하기 전의 argmax입니다. fold는 oof_v4.npz의 fold(= 제출 커널의 분할)를 씁니다.

사용: python analysis_confusion/plot_confusion.py    # 그림은 analysis_confusion/figures/, 수치는 표준 출력
"""
import json, os, tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
ORGS = [("GPT", "gpt"), ("Gemini", "gemini"), ("Claude", "claude"), ("Qwen", "qwen"), ("GLM", "glm"), ("Kimi", "kimi")]
MULTI = [("GPT", "gpt"), ("Gemini", "gemini"), ("Claude", "claude"), ("GLM", "glm")]  # 모델이 둘 이상인 회사

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUES = LinearSegmentedColormap.from_list(
    "seq", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
_installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
FONTS = [f for f in ("Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic")
         if f in _installed] + ["DejaVu Sans"]  # 한글 글꼴은 설치된 것만 씁니다
plt.rcParams.update({"font.family": FONTS,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 10,
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": INK, "ytick.color": INK})


def v4_pred():
    d = np.load(ROOT / "analysis_v4/oof_v4.npz", allow_pickle=True)
    return d["y"], d["fold"], d["eval_lp"].argmax(1), [str(l) for l in d["labels"]]


def v7_pred(y, fold, labels):
    """v7 노트북의 블렌드 셀과 게이트 셀을 저장된 OOF 로짓으로 실행해 (블렌드 예측, 게이트 뒤 예측)을 돌려줍니다."""
    import pandas as pd
    d = np.load(ROOT / "analysis_v7/oof_v7.npz", allow_pickle=True)
    assert (d["y"] == y).all(), "oof_v7.npz의 지문 순서가 oof_v4.npz와 다릅니다."
    os.environ.update(NTL_WORK=tempfile.mkdtemp(), NTL_CFG="{}", MPLBACKEND="Agg")
    nb = json.load(open(ROOT / "name_that_llm_v7.ipynb"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    find = lambda key: next(s for s in code if key in s)
    g = {}
    exec(find("T0 = time.time()"), g)  # CFG, LABELS, L2I
    assert g["LABELS"] == labels
    g.update(y=y, oof_ng=d["ng_logits"], lora_oof=d["lora_logits"], deb_oof=d["deb_logits"],
             train=pd.DataFrame({"id": d["ids"]}), test=pd.DataFrame({"id": d["ids"][:3]}))
    g["cv_folds"] = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(g["CFG"]["n_folds"])]
    g["holdout_idx"] = g["cv_folds"][0][1]
    # 테스트 로짓은 없으므로 OOF 앞 3행을 자리 채움으로 넣습니다(테스트 쪽 출력은 의미 없음).
    g.update(ng_test=d["ng_logits"][:3], lora_test=d["lora_logits"][:3], deb_test=d["deb_logits"][:3])
    exec(find("def log_softmax_np"), g)  # 5장: 온도, 3분기 블렌드
    blend = g["eval_lp"].argmax(1)
    exec(find("FAMILIES = "), g)         # 5-1장: 회사 게이트
    return blend, g["eval_lp"].argmax(1)


def org_matrix(y, p, labels, prefix):
    idx = [i for i, l in enumerate(labels) if l.startswith(prefix)]
    n = len(idx)
    cm = np.zeros((n, n + 1))
    for r, k in enumerate(idx):
        pred = p[y == k]
        for c, j in enumerate(idx):
            cm[r, c] = (pred == j).mean()
        cm[r, n] = 1 - cm[r, :n].sum()
    short = [labels[k].split("-", 1)[1] if "-" in labels[k] else labels[k] for k in idx]
    return cm, short


def draw(ax, cm, short, title):
    n = len(short)
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1, aspect="equal")
    ax.axvline(n - 0.5, color=SURFACE, lw=4)  # 외부 열을 따로 떼어 보이게
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=10,
                fontweight="bold" if r == c else "normal", color=SURFACE if v > 0.5 else INK)
    for d in range(n):
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False, ec=INK, lw=1.5))
    ax.set_xticks(range(n + 1), short + ["외부"], rotation=30, ha="right")
    ax.set_yticks(range(n), short)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, pad=8)


def plot_org(y, preds, labels, org, prefix, out):
    mats = [(name, *org_matrix(y, p, labels, prefix)) for name, p in preds]
    n = len(mats[0][2])
    fig, axes = plt.subplots(1, len(mats), figsize=(len(mats) * (0.9 * (n + 1) + 2.1), 0.9 * n + 2.3))
    for ax, (name, cm, short) in zip(axes, mats):
        draw(ax, cm, short, name)
    top = 1 - 0.75 / fig.get_figheight()  # 제목 두 줄의 높이를 인치로 고정
    fig.suptitle(f"{org} 내부 혼동 행렬", x=0.02, y=1 - 0.12 / fig.get_figheight(), ha="left", va="top", fontsize=13)
    fig.text(0.02, 1 - 0.50 / fig.get_figheight(), "OOF 지문 2,699개 · 행 = 정답, 열 = 예측, 값 = 행 비율 · 외부 = 다른 회사 모델로 예측",
             fontsize=9, color=MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.2)  # 기울인 열 이름이 잘리지 않게
    plt.close(fig)


def report(name, y, p, labels):
    org = np.array([next(n for n, pre in ORGS if l.startswith(pre)) for l in labels])
    yo, po = org[y], org[p]
    wrong = p != y
    same = wrong & (yo == po)
    print(f"\n[{name}] BA {balanced_accuracy_score(y, p):.4f} | 오답 {wrong.sum()}개, 그중 같은 회사 {same.sum()}개 "
          f"({same.sum() / wrong.sum():.1%})")
    for o, _ in ORGS:
        m = yo == o
        hit = m & (po == o)
        print(f"  {o:<7} 회사를 맞힌 비율 {hit.sum() / m.sum():.3f} | 맞혔을 때 모델까지 맞힌 비율 {(p[hit] == y[hit]).mean():.3f} "
              f"| 지문 {m.sum()}개")
    print("  모델별 recall: " + ", ".join(f"{l} {(p[y == k] == k).mean():.3f}" for k, l in enumerate(labels)))
    pairs = sorted(((np.mean(p[y == a] == b), labels[a], labels[b]) for a in range(len(labels)) for b in range(len(labels))
                    if a != b), reverse=True)[:12]
    print("  많이 헷갈린 쌍 (정답 → 예측): " + ", ".join(f"{a} → {b} {v:.3f}" for v, a, b in pairs))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    y, fold, p4, labels = v4_pred()
    p7_blend, p7 = v7_pred(y, fold, labels)
    print(f"\nv7 재현 확인: 블렌드 OOF BA {balanced_accuracy_score(y, p7_blend):.4f} -> 게이트 {balanced_accuracy_score(y, p7):.4f} "
          "(제출 커널 로그: 0.5868 -> 0.5882)")
    preds = [("v4 · n-gram + LoRA", p4), ("v7 · 3분기 블렌드 + 회사 게이트", p7)]
    for org, prefix in MULTI:
        plot_org(y, preds, labels, org, prefix, OUT / f"cm_{prefix}.png")
        print("저장:", (OUT / f"cm_{prefix}.png").relative_to(ROOT))
    report("v4", y, p4, labels)
    report("v7", y, p7, labels)
    for org, prefix in MULTI:
        for name, p in preds:
            cm, short = org_matrix(y, p, labels, prefix)
            print(f"\n{org} / {name}  (열: {', '.join(short)}, 외부)")
            for s, row in zip(short, cm):
                print(f"  {s:<16}" + " ".join(f"{v:.3f}" for v in row))
